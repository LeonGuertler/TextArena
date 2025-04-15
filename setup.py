from setuptools import setup, find_packages

setup(
    name="textarena",
    version="0.6.4",
    url="https://github.com/LeonGuertler/TextArena",
    author="Leon Guertler",
    author_email="Guertlerlo@cfar.a-star.edu.sg",
    description="A Collection of Competitive Text-Based Games for Language Model Evaluation and Reinforcement Learning",
    packages=find_packages(include=["textarena", "textarena.*"]),
    include_package_data=True,
    package_data={
        "": ["*.json"],  # Include all JSON files in any package
        "": ["*.jsonl"],  # Include all JSON files in any package
        "textarena.envs.two_player.TruthAndDeception": ["*.json"],  # Explicitly include JSON files in this directory
        "textarena.envs.two_player.Debate": ["*.json"],  # Explicitly include JSON files in this directory
        "textarena": ["envs/**/*.json"],  # Recursive include from textarena root
        "textarena.envs.two_player.Debate": ["topics.json"],
    },
    install_requires=[
        "requests",
        "openai",
        "rich",
        "nltk",
        "chess",
        "networkx",
        "python-dotenv",
        "websockets",
    ],
    python_requires='>=3.10',
)
