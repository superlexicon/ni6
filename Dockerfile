# IM-OSINT Application Dockerfile (CUDA)
# Requires base image: 10.0.0.14:5000/im-osint-base-cuda:latest
# Build: docker build -t 10.0.0.14:5000/im-osint:latest .

FROM 10.0.0.14:5000/im-osint-base-cuda:latest

WORKDIR /app

# Copy application code only (dependencies and weights are in base image)
COPY --exclude=photoholmes/weights app/ /app/app/
COPY .env* /app/

# Expose port
EXPOSE 12410

# Default command
CMD ["python3", "-m", "app.main"]
