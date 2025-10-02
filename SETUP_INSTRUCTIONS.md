# TextArena Project Setup Instructions

## Overview
This project is a text-based game environment framework featuring a Vending Machine simulation game. The simulation includes supply chain management with inventory control, demand forecasting, and profit optimization.

## Prerequisites
- Python 3.10 or higher
- `uv` package manager (recommended)
- OpenAI API key

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd TextArena
```

### 2. Install Dependencies
Install the project dependencies using uv:
```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
```

### 3. Set Environment Variables
Before running the project, you need to set your OpenAI API key:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 4. Run the Demo
Execute the vending machine simulation:
```bash
uv run python examples/vending_offline_demo.py
```

## Project Structure

The project consists of:
- **VM Agent (Vending Machine Controller)**: Manages inventory, places orders, and sets prices
- **Demand Agent (Customer Simulation)**: Simulates customer purchasing behavior

## Game Features

The simulation includes:
- Multi-item inventory management
- Lead times for order fulfillment
- Holding costs for inventory
- News events affecting demand
- Profit optimization strategies

## Customization

You can modify the `examples/vending_offline_demo.py` file to customize:
- Item types and pricing
- News events
- Game duration
- Agent behavior patterns

After making changes, run:
```bash
uv run python examples/vending_offline_demo.py
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure `OPENAI_API_KEY` is correctly set
   - Verify your API key is valid and has sufficient credits

2. **Dependency Issues**:
   - Make sure you're using Python 3.10+
   - Try reinstalling dependencies: `uv sync --reinstall`

3. **Permission Issues**:
   - Ensure you have internet connectivity for OpenAI API access
   - Check firewall settings


