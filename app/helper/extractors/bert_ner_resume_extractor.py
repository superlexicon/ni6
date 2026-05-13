"""
BERT NER-based Resume Extractor

This module provides Named Entity Recognition based resume extraction
using the BERT model for improved accuracy over regex-based approaches.
"""

import logging
import re
from typing import List, Dict, Optional, Any, Tuple

from app.schemas.resume_schema import ResumeData
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.core.bert_ner_model import BertNerModel, get_bert_ner_model
from app.core.logger import get_logger

logger = get_logger()


class BertNerResumeExtractor:
    """
    Extract resume data using BERT-based Named Entity Recognition.

    This extractor uses a pre-trained BERT model fine-tuned on resume data
    to identify entities like names, emails, phone numbers, skills, education,
    work experience, and more.
    """

    # Entity types from the model
    ENTITY_TYPES = {
        # Contact info
        'EMAIL': 'email',
        'PHONE': 'phone_number',
        'NAME': 'full_name',
        'LOCATION': 'address',

        # Professional info
        'DESIGNATION': 'professional_title',
        'ORG': 'organization',  # Companies/employers
        'DATE': 'date',  # Employment/education dates
        'SKILL': 'skill',

        # Education
        'EDUCATION_DEGREE': 'degree',
        'EDUCATION_INSTITUTION': 'institution',
    }

    def __init__(self):
        self.logger = logger
        self.text_extractor = DocumentTextExtractor()
        self._model_wrapper: Optional[BertNerModel] = None
        self._label_map: Optional[Dict[int, str]] = None

    @property
    def model_wrapper(self) -> BertNerModel:
        """Lazy load the BERT NER model"""
        if self._model_wrapper is None:
            self._model_wrapper = get_bert_ner_model()
        return self._model_wrapper

    @property
    def label_map(self) -> Dict[int, str]:
        """Lazy load the label map"""
        if self._label_map is None:
            self._label_map = self.model_wrapper.get_label_map()
        return self._label_map

    async def extract(self, content: bytes, is_pdf: bool = True) -> ResumeData:
        """
        Extract all standard resume/CV fields using BERT NER.

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            ResumeData with extracted fields and confidence scores
        """
        try:
            # Extract text using DocTr
            text = await self.text_extractor.extract_text(content, is_pdf=is_pdf)

            self.logger.info(f"Extracted {len(text)} characters from document")

            # Store the raw text for later use
            self._raw_text = text

            # Get BERT NER model
            model, tokenizer = await self.model_wrapper.get_model_with_gpu()

            # Run NER extraction
            entities = await self._extract_entities(text, model, tokenizer)

            # Map to ResumeData schema
            resume_data = self._map_to_resume_data(entities)

            # Store raw OCR text for debugging/auditing
            resume_data.raw_data = self._raw_text

            self.logger.info(f"Successfully extracted resume data with {len(resume_data.confidence_scores)} fields")
            return resume_data

        except Exception as e:
            self.logger.error(f"BERT NER resume extraction failed: {str(e)}")
            return ResumeData(confidence_scores={})

    async def _extract_entities(
        self,
        text: str,
        model,
        tokenizer
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using BERT NER model.

        Args:
            text: Input text
            model: BERT NER model
            tokenizer: BERT tokenizer

        Returns:
            List of extracted entities with labels, spans, and confidence scores
        """
        try:
            import torch

            # Get device from model
            device = next(model.parameters()).device

            # Tokenize input
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_offsets_mapping=True
            )

            # Move to device
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            offset_mapping = tokens['offset_mapping']

            # Run inference
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Get confidence scores
            probs = torch.softmax(logits, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]

            # Process predictions
            entities = []
            current_entity = None

            for i, (pred_id, conf, offset) in enumerate(zip(
                predictions[0].cpu().numpy(),
                confidences[0].cpu().numpy(),
                offset_mapping.cpu().numpy()
            )):
                # Skip special tokens and padding
                if offset[0] == 0 and offset[1] == 0:
                    continue

                # Get label
                label = self.label_map.get(int(pred_id), "O")

                # Get text span
                start, end = offset[0], offset[1]
                if start >= end or start >= len(text) or end > len(text):
                    continue

                span_text = text[start:end]

                # Process BIO tagging
                if label == "O":
                    # Outside any entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                elif label.startswith("B-"):
                    # Beginning of entity
                    if current_entity:
                        entities.append(current_entity)

                    entity_type = label[2:]
                    current_entity = {
                        'type': entity_type,
                        'text': span_text,
                        'start': start,
                        'end': end,
                        'confidence': float(conf),
                        'tokens': [span_text]
                    }
                elif label.startswith("I-"):
                    # Inside/continuation of entity
                    entity_type = label[2:]
                    if current_entity and current_entity['type'] == entity_type:
                        # Continue current entity
                        current_entity['text'] += span_text
                        current_entity['end'] = end
                        current_entity['tokens'].append(span_text)
                        # Update confidence to average
                        current_entity['confidence'] = (
                            (current_entity['confidence'] * (len(current_entity['tokens']) - 1) + conf)
                            / len(current_entity['tokens'])
                        )
                    else:
                        # New entity without B- tag
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            'type': entity_type,
                            'text': span_text,
                            'start': start,
                            'end': end,
                            'confidence': float(conf),
                            'tokens': [span_text]
                        }

            # Don't forget the last entity
            if current_entity:
                entities.append(current_entity)

            self.logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            return []

    def _map_to_resume_data(self, entities: List[Dict[str, Any]]) -> ResumeData:
        """
        Map extracted entities to ResumeData schema.

        Args:
            entities: List of extracted entities

        Returns:
            ResumeData with populated fields
        """
        resume_data = ResumeData()

        # Group entities by type
        entities_by_type: Dict[str, List[Dict]] = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Map entities to resume fields

        # Full name - use the first detected name, usually at the top
        if 'NAME' in entities_by_type and entities_by_type['NAME']:
            # Sort by position (top of page first)
            name_entities = sorted(entities_by_type['NAME'], key=lambda x: x['start'])
            resume_data.full_name = name_entities[0]['text'].strip()
            resume_data.confidence_scores['full_name'] = name_entities[0]['confidence'] * 100

        # Email
        if 'EMAIL' in entities_by_type and entities_by_type['EMAIL']:
            email = entities_by_type['EMAIL'][0]['text'].strip()
            # Validate email format
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                resume_data.email = email
                resume_data.confidence_scores['email'] = entities_by_type['EMAIL'][0]['confidence'] * 100

        # Phone number
        if 'PHONE' in entities_by_type and entities_by_type['PHONE']:
            resume_data.phone_number = entities_by_type['PHONE'][0]['text'].strip()
            resume_data.confidence_scores['phone_number'] = entities_by_type['PHONE'][0]['confidence'] * 100

        # Address/Location
        if 'LOCATION' in entities_by_type and entities_by_type['LOCATION']:
            # Combine multiple locations
            locations = [e['text'].strip() for e in entities_by_type['LOCATION']]
            resume_data.address = ', '.join(locations[:3])  # Limit to first 3
            avg_conf = sum(e['confidence'] for e in entities_by_type['LOCATION'][:3]) / min(3, len(entities_by_type['LOCATION']))
            resume_data.confidence_scores['address'] = avg_conf * 100

        # Professional title/Designation
        if 'DESIGNATION' in entities_by_type and entities_by_type['DESIGNATION']:
            # Use the most confident designation
            designation = max(entities_by_type['DESIGNATION'], key=lambda x: x['confidence'])
            resume_data.professional_title = designation['text'].strip()
            resume_data.confidence_scores['professional_title'] = designation['confidence'] * 100

        # Skills
        if 'SKILL' in entities_by_type and entities_by_type['SKILL']:
            skills = list(set(e['text'].strip() for e in entities_by_type['SKILL']))
            resume_data.skills = skills
            avg_conf = sum(e['confidence'] for e in entities_by_type['SKILL']) / len(entities_by_type['SKILL'])
            resume_data.confidence_scores['skills'] = avg_conf * 100

        # Organizations (Companies/ Employers)
        orgs = []
        if 'ORG' in entities_by_type and entities_by_type['ORG']:
            orgs = [e['text'].strip() for e in entities_by_type['ORG']]

        # Education entries
        education_entries = []
        degrees = entities_by_type.get('EDUCATION_DEGREE', [])
        institutions = entities_by_type.get('EDUCATION_INSTITUTION', [])
        dates = entities_by_type.get('DATE', [])

        # Match degrees with institutions
        for degree in degrees:
            entry = {'degree': degree['text'].strip()}

            # Find nearest institution (within 200 characters)
            nearest_inst = None
            min_dist = 200
            for inst in institutions:
                dist = abs(inst['start'] - degree['start'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_inst = inst

            if nearest_inst:
                entry['institution'] = nearest_inst['text'].strip()

            # Find nearest date
            nearest_date = None
            min_dist = 200
            for date in dates:
                dist = abs(date['start'] - degree['start'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_date = date

            if nearest_date:
                entry['year'] = nearest_date['text'].strip()

            education_entries.append(entry)

        if education_entries:
            resume_data.education_entries = education_entries
            avg_conf = sum(
                sum(e.get('confidence', 0.8) for e in degrees + institutions + dates) /
                max(1, len(degrees) + len(institutions) + len(dates))
            )
            resume_data.confidence_scores['education_entries'] = avg_conf * 100

        # Work experience entries
        work_experience_entries = []
        for org in orgs:
            entry = {'company': org['text']}

            # Find nearest designation
            nearest_desig = None
            min_dist = 200
            for desig in entities_by_type.get('DESIGNATION', []):
                dist = abs(desig['start'] - org['start'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_desig = desig

            if nearest_desig:
                entry['position'] = nearest_desig['text'].strip()

            # Find nearest dates
            nearest_dates = sorted(
                [d for d in dates if abs(d['start'] - org['start']) < 300],
                key=lambda x: abs(x['start'] - org['start'])
            )[:2]

            if len(nearest_dates) >= 1:
                entry['start_year'] = nearest_dates[0]['text'].strip()
            if len(nearest_dates) >= 2:
                entry['end_year'] = nearest_dates[1]['text'].strip()

            work_experience_entries.append(entry)

        if work_experience_entries:
            resume_data.work_experience_entries = work_experience_entries
            # Use organization confidence for work experience
            avg_conf = sum(e['confidence'] for e in entities_by_type.get('ORG', [])) / max(1, len(entities_by_type.get('ORG', [])))
            resume_data.confidence_scores['work_experience_entries'] = avg_conf * 100

        # Extract LinkedIn URL (post-processing with regex since BERT NER may not catch it)
        full_text = ' '.join(e['text'] for e in entities)
        linkedin_match = re.search(
            r'(https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+)',
            full_text,
            re.IGNORECASE
        )
        if linkedin_match:
            resume_data.linkedin_url = linkedin_match.group(1)
            resume_data.confidence_scores['linkedin_url'] = 95.0

        return resume_data

    async def extract_with_details(
        self,
        content: bytes,
        is_pdf: bool = True
    ) -> Tuple[ResumeData, List[Dict[str, Any]]]:
        """
        Extract resume data and return both the schema and raw entities.

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            Tuple of (ResumeData, raw_entities)
        """
        try:
            # Extract text
            text = await self.text_extractor.extract_text(content, is_pdf=is_pdf)

            # Get BERT NER model
            model, tokenizer = await self.model_wrapper.get_model_with_gpu()

            # Run NER extraction
            entities = await self._extract_entities(text, model, tokenizer)

            # Map to ResumeData schema
            resume_data = self._map_to_resume_data(entities)

            # Store raw OCR text for debugging/auditing
            resume_data.raw_data = text

            return resume_data, entities

        except Exception as e:
            self.logger.error(f"BERT NER resume extraction with details failed: {str(e)}")
            return ResumeData(confidence_scores={}), []
