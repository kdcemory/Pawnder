# Pawnder
# Pawnder

AI-powered dog behavior analysis application that detects emotional states from photos and videos.

## Project Overview

Pawnder is an innovative application that uses deep learning to analyze dog behavior from photos and videos, providing emotional state assessment based on the Primary Behavior Matrix. The app helps dog owners, trainers, and veterinarians better understand canine behavior through AI-powered analysis of key behavioral indicators.

### Key Features

- **Behavior Analysis**: Upload photos or videos of dogs to receive instant analysis
- **Emotional State Assessment**: Get detailed analysis of your dog's emotional state (happy, relaxed, fearful, etc.)
- **Confidence Scoring**: Every assessment includes confidence scores for transparency
- **Educational Content**: Learn about dog body language and behavior patterns
- **User History**: Track your dog's behavioral patterns over time

## System Architecture

The Pawnder application follows a modern, scalable architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │   Backend       │     │   ML Platform   │
│                 │     │                 │     │                 │
│  - Mobile App   │     │  - REST API     │     │  - Model Train  │
│  - Web App      │◄───►│  - Services     │◄───►│  - Inference    │
│  - Admin Panel  │     │  - Auth         │     │  - Evaluation   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               ▲
                               │
                        ┌──────┴──────┐
                        │  Database   │
                        │    & S3     │
                        └─────────────┘
```

### Technology Stack

- **Frontend**: 
  - Mobile: Flutter
  - Web: React, TypeScript
  
- **Backend**:
  - API: FastAPI
  - Services: Python microservices
  - Authentication: AWS Cognito
  
- **Machine Learning**:
  - Framework: PyTorch/TensorFlow
  - Model Serving: AWS Lambda + Docker
  
- **Infrastructure**:
  - Cloud: AWS (S3, Lambda, EC2, API Gateway)
  - CI/CD: GitHub Actions
  - Monitoring: AWS CloudWatch

## Setup Instructions

### Prerequisites

- Git
- Python 3.10+
- Node.js 16+
- Flutter SDK
- Docker and Docker Compose
- AWS CLI (configured)

### Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/your-username/pawnder.git
cd pawnder
```

2. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Backend setup**

```bash
cd backend/api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

4. **Frontend setup (Web)**

```bash
cd frontend/web
npm install
npm start
```

5. **Frontend setup (Mobile)**

```bash
cd frontend/mobile
flutter pub get
flutter run
```

6. **ML environment setup**

```bash
cd ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

7. **Using Docker (optional)**

```bash
docker-compose up
```

## Development Workflow

1. **Creating a new feature**

```bash
git checkout -b feature/your-feature-name
# Make your changes
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Create a Pull Request on GitHub
```

2. **Workflow steps**

- Create an issue describing the feature or bug
- Create a branch linked to the issue
- Implement the changes with appropriate tests
- Submit a pull request for review
- After approval, merge to develop/main branch

3. **Code quality**

- All code must pass linting checks
- Unit tests are required for new features
- Code reviews are mandatory before merging

## Directory Structure

```
pawnder/
├── frontend/                # Frontend applications
│   ├── mobile/              # Flutter mobile app
│   └── web/                 # React web application
├── backend/                 # Backend services
│   ├── api/                 # FastAPI main application
│   └── services/            # Microservices
│       ├── media/           # Media processing service
│       ├── analysis/        # Analysis service
│       └── auth/            # Authentication service
├── ml/                      # Machine learning components
│   ├── data/                # Data management
│   ├── models/              # Model definitions
│   ├── training/            # Training scripts
│   └── evaluation/          # Evaluation tools
├── docs/                    # Documentation
├── infrastructure/          # Infrastructure as code
│   ├── terraform/           # Terraform configurations
│   └── docker/              # Docker configurations
└── .github/                 # GitHub workflows
```

## Contribution Guidelines

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

### Coding Standards

- Follow PEP 8 for Python code
- Use ESLint rules for JavaScript/TypeScript
- Follow Dart analysis rules for Flutter code
- Write meaningful commit messages
- Document all functions and classes

### Pull Request Process

1. Ensure all tests pass in CI
2. Update documentation if needed
3. Link the PR to relevant issues
4. Get at least one code review approval
5. Maintainers will merge the PR when ready

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford Dog Dataset for providing training data
- Primary Behavior Matrix for the behavior classification framework
- All contributors who have helped shape this project
