# Third-Party Licenses

NI6 incorporates several third-party components with their own licenses. This document provides transparency about these licenses.

## PhotoHolmes Components

NI6 uses the PhotoHolmes framework for image forgery detection. The following methods have specific license requirements:

| Component | License | Compatibility with GPL-3.0 |
|-----------|---------|---------------------------|
| **NoiseSniffer** | GPL-3.0 | Compatible (same license) |
| **ZERO** | AGPL-3.0 | Compatible (AGPL-3.0 is GPL-3.0 + network use requirement) |
| **CAT-Net** | Non-commercial research use only | Not compatible with commercial use |
| **SpliceBuster** | Non-commercial research use only | Not compatible with commercial use |
| **TruFor** | Non-commercial research use only | Not compatible with commercial use |
| **Focal** | MIT | Compatible (permissive) |
| **PSCCNet** | MIT | Compatible (permissive) |
| **EXIF as Language** | MIT | Compatible (permissive) |

## License Implications

### For GPL-3.0 Components (NoiseSniffer, ZERO)

These components are fully compatible with NI6's GPL-3.0 license. Any derivative works must also be licensed under GPL-3.0.

### For MIT-Licensed Components (Focal, PSCCNet, EXIF as Language)

These permissively licensed components can be used within GPL-3.0 works without issue.

### For Non-Commercial Research-Only Components (CAT-Net, SpliceBuster, TruFor)

**Important**: These components have additional restrictions:
- Use is limited to non-commercial purposes
- Use is limited to academic/research purposes
- Commercial use requires separate licensing from the original authors

If you intend to use NI6 for commercial purposes, you must either:
1. Remove these components from your build, or
2. Obtain appropriate commercial licenses from the original component authors

## Other Major Dependencies

| Component | License | Notes |
|-----------|---------|-------|
| DeepFace | MIT | Compatible |
| MediaPipe | Apache-2.0 | Compatible with GPL-3.0 |
| DoTR (OCR) | Apache-2.0 | Compatible with GPL-3.0 |
| PyTorch | BSD-style | Compatible |
| TensorFlow | Apache-2.0 | Compatible with GPL-3.0 |

## Recommendations for Users

1. **Academic/Research Use**: All PhotoHolmes methods can be used under fair use/research exceptions
2. **Commercial Use**: Consider disabling CAT-Net, SpliceBuster, and TruFor methods unless you obtain commercial licenses
3. **Derivative Works**: Any modifications to NI6 must be released under GPL-3.0

## References

- [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)
- [PhotoHolmes Repository](https://github.com/thefcraft/PhotoHolmes)
- [AGPL v3](https://www.gnu.org/licenses/agpl-3.0.html)
- [MIT License](https://opensource.org/licenses/MIT)

For specific licensing questions regarding the non-commercial components, please contact the original authors of those components.
