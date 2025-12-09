# SoMe: A Realistic Benchmark for LLM-based Social Media Agents

## Overview

SoMe (Social Media) is a comprehensive benchmark designed to evaluate the capabilities of Large Language Model (LLM)-based agents in realistic social media scenarios. This benchmark provides a standardized framework for testing and comparing social media agents across multiple dimensions of performance.

![SoMe Benchmark Overview](https://github.com/livxue/SoMe/tree/main/pics/framework.png)


## Features

SoMe benchmark includes evaluation of social media agents across the following key tasks:

- **Realtime Event Detection** - Assessing how well agents can identify and track emerging events in real-time
- **Streaming Event Summary** - Evaluating the agent's capacity to summarize ongoing events from streaming data
- **Misinformation Detection** - Testing the agent's capability to identify and flag potentially false or misleading information
- **User Behavior Prediction** - Evaluating how well agents can predict user interactions with social media content
- **User Emotion Analysis** - Assessing the agent's ability to analyze user emotions towards social media content
- **User Comment Simulation** - Testing how realistically agents can simulate user comments
- **Media Content Recommendation** - Evaluating an agent's ability to recommend relevant media content to users based on their interests and preferences
- **Social Media Question Answering** - Measuring the agent's ability to accurately answer questions about social media content

## Dataset Statistics

The SoMe benchmark includes comprehensive datasets for each task, with the following statistics:

| Task | #Query | #Data | Data Type |
|------|-------------|-----------|----------|
| Real-time Event Detection | 568 | 476,611 | Posts |
| Streaming Event Summary | 154 | 7,898,959 | Posts |
| Misinformation Detection | 1,451 | 27,137 | Posts & Knowledge |
| User Behavior Prediction | 3,000 | 840,200 | Posts & Users|
| User Emotion Analysis | 2,696 | 840,200 | Posts & Users |
| User Comment Simulation | 4,000 | 840,200 | Posts & Users |
| Media Content Recommendation | 4,000 | 840,200 | Posts & Users |
| Social Media Question Answering | 2,000 | 8,651,759 | Posts & Users |
| Total | 17,869 | 9,242,907 | All |

## Project Structure

```
Social-Media-Agent/
├── agent.py                    # Main social media agent implementation
├── tasks/                      # Task-specific modules
│   ├── media_content_recommend/
│   ├── misinformation_detection/
│   ├── realtime_event_detection/
│   ├── social_media_question_answering/
│   ├── streaming_event_summary/
│   ├── user_comment_simulation/
│   ├── user_emotion_analysis/
│   └── user_behavior_prediction/
├── tools/                      # Tools for social media analysis
├── test_*.py                   # Test scripts for each task
├── eval_scripts/               # Evaluation scripts for scoring
├── results/                    # Directory for storing results
├── datasets/                   # Dataset directory
└── database/                   # Database directory
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Social-Media-Agent.git
cd Social-Media-Agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the qwen-agent package:
```bash
pip install -e ./qwen_agent
```

## Usage

### Running Individual Tasks

Each task can be evaluated using its corresponding test script:

```bash
# Realtime Event Detection
python test_realtime_event_detection.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# Streaming Event Summary
python test_streaming_event_summary.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# Misinformation Detection
python test_misinformation_detection.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# User bahavior Prediction
python test_user_behavior_prediction.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# User Emotion Analysis
python test_user_emotion_analysis.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# User Comment Simulation
python test_user_comment_simulation.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# Media Content Recommendation
python test_media_content_recommend.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# Social Media Question Answering
python test_social_media_question_answering.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

```

### Command Line Arguments

- `--model`: The model name to use (e.g., "deepseek-chat")
- `--base_url`: The base URL for the model server (e.g., "https://api.deepseek.com")
- `--api_key`: The API key for the model server
- `--output_path`: The output path for results (default: "results/[task_name]")

### Evaluation

After running the test scripts, you can evaluate the results using the provided evaluation scripts:

```bash
# For each task, use the corresponding evaluation script
python eval_scripts/[TASK]_extraction.py
python eval_scripts/[TASK]_compute_score.py
# Or
python eval_scripts/[TASK]_scoring.py
python eval_scripts/[TASK]_compute_score.py
```
***Note***: The LLM setting of evaluation is in `eval_scripts/settings.json`

## Model Support

The benchmark supports various LLM models including:

- Qwen series models
- OpenAI compatible models
- Any model that provides an OpenAI-compatible API endpoint

## Citation

If you use this benchmark in your research, please cite:

```
@inproceedings{some2026,
  title={SoMe: A Realistic Benchmark for LLM-based Social Media Agents},
  author={Dizhan Xue, Jing Cui, Shengsheng Qian, Chuanrui Hu, Changsheng Xu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Contributing

We welcome contributions to improve the benchmark! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This benchmark is built upon the Qwen-Agent framework. We thank the Qwen team for their excellent work on developing this framework.