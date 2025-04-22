# Stratego AI Bedrock Agents Battle

A project that simulates Stratego games between two AWS Bedrock agents (Claude 3.5 Sonnet and Claude 3.7 Sonnet) using the TextArena library.

## Prerequisites

Before you begin, ensure you have:

* Python 3.x (Recommended: Python 3.8 or later)
* AWS Account with access to AWS Bedrock
* AWS CLI installed and configured (Recommended)
* Boto3 library (AWS SDK for Python)
* TextArena library

## Installation

Install the required dependencies:

```bash
pip install boto3 textarena
```

## Configuration
Setting Up AWS Credentials
You have two options for configuring AWS credentials:

### Option 1: Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### Option 2: AWS CLI

```bash
aws configure
```

## Usage

1. Set your AWS Region

```bash
export AWS_REGION="us-west-2"
```

2. Run your sample code

Make sure you are in the right directory:

```bash
python sample.py
```

