<div align="center">
<img src="https://github.com/LivXue/SoMe/blob/main/pics/logo.png" alt="SoMe Benchmark Logo" width="33%">
</div>

# 🤖 SoMe: A Realistic Benchmark for LLM-based Social Media Agents

<div align="center" style="line-height: 1.5;">

[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/LivXue/SoMe)
[![Dataset](https://img.shields.io/badge/Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/LivXue/Social-Media-Agents-Benchmark)
[![Paper](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2512.14720)
[![AAAI](https://img.shields.io/badge/AAAI-0066cc?style=for-the-badge&logo=readthedocs&logoColor=white)](https://ojs.aaai.org/index.php/AAAI/article/view/37113)

</div>

---

## 📋 Overview

SoMe is a comprehensive benchmark designed to evaluate the capabilities of Large Language Model (LLM)-based agents in realistic social media scenarios. This benchmark provides a standardized framework for testing and comparing social media agents across multiple dimensions of performance.

SoMe comprises a diverse collection of:
- **8 social media agent tasks**
- **9,164,284 posts** from various social media platforms
- **6,591 user profiles** with rich behavioral data
- **25,686 reports** from external websites
- **17,869 meticulously annotated task queries**

![SoMe Benchmark Overview](https://github.com/LivXue/SoMe/blob/main/pics/framework.png)

*Figure 1: An example of agentic task in SoMe*

---

## 📰 News

- **[2026.01]** 📧 **We release [MailMind: An AI-powered Email System that Can Do Your Job](https://github.com/LivXue/open-email-agent)**
- **[2025.11]** 🎉 Our paper is accepted by AAAI 2026!

---

## ✨ Features

SoMe benchmark evaluates social media agents across 8 key tasks, covering diverse aspects of social media intelligence:

| Task Category | Task Name | Description |
|---------------|-----------|-------------|
| **Post-centered** | 🚨 Realtime Event Detection (RED) | Identify and track emerging events in real-time |
| **Post-centered** | 📊 Streaming Event Summary (SES) | Summarize ongoing events from streaming data |
| **Post-centered** | 🚫 Misinformation Detection (MID) | Identify and flag potentially false or misleading information |
| **User-centered** | 🎯 User Behavior Prediction (UBP) | Predict user interactions with social media content |
| **User-centered** | 😊 User Emotion Analysis (UEA) | Analyze user emotions towards social media content |
| **User-centered** | 💬 User Comment Simulation (UCS) | Simulate realistic user comments |
| **Comprehensive** | 📱 Media Content Recommendation (MCR) | Recommend relevant media content based on user interests |
| **Comprehensive** | ❓ Social Media Question-Answering (SMQ) | Accurately answer questions about social media content |

---

## 📈 Dataset Statistics

The SoMe benchmark includes comprehensive datasets for each task, with the following statistics:

| Task | # Query | # Data | Data Type |
|------|---------|--------|-----------|
| 🚨 Real-time Event Detection | 568 | 476,611 | Posts |
| 📊 Streaming Event Summary | 154 | 7,898,959 | Posts |
| 🚫 Misinformation Detection | 1,451 | 27,137 | Posts & Knowledge |
| 🎯 User Behavior Prediction | 3,000 | 840,200 | Posts & Users |
| 😊 User Emotion Analysis | 2,696 | 840,200 | Posts & Users |
| 💬 User Comment Simulation | 4,000 | 840,200 | Posts & Users |
| 📱 Media Content Recommendation | 4,000 | 840,200 | Posts & Users |
| ❓ Social Media Question-Answering | 2,000 | 8,651,759 | Posts & Users |
| **Total** | **17,869** | **9,242,907** | **All** |

---

## 🏆 Evaluation Results

We evaluated various agentic LLMs on the SoMe benchmark. Below are the comprehensive evaluation results across all 8 tasks:

![SoMe Benchmark Results](https://github.com/LivXue/SoMe/blob/main/pics/result.png)

*Figure 2: Performance comparison of different agentic models across SoMe benchmark tasks*

---

## 📁 Project Structure

```
Social-Media-Agent/
├── 🤖 agent.py                    # Main social media agent implementation
├── 🔧 qwen_agent/                 # Qwen-Agent library
├── 📋 tasks/                      # Task-specific modules
│   ├── 📱 media_content_recommend/
│   ├── 🚫 misinformation_detection/
│   ├── 🚨 realtime_event_detection/
│   ├── ❓ social_media_question_answering/
│   ├── 📊 streaming_event_summary/
│   ├── 💬 user_comment_simulation/
│   ├── 😊 user_emotion_analysis/
│   └── 🎯 user_behavior_prediction/
├── 🛠️ tools/                      # Tools for social media analysis
├── 🧪 test_*.py                   # Test scripts for each task
├── 📊 eval_scripts/               # Evaluation scripts for scoring
├── 📂 results/                    # Directory for storing results
├── 📊 datasets/                   # Dataset directory
└── 💾 database/                   # Database directory
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+ installed on your system
- Git installed for repository cloning
- Sufficient disk space for data (recommended: 50GB+)

### Installation Steps

1. **📥 Clone the repository**
   ```bash
   git clone https://github.com/LivXue/SoMe.git
   cd SoMe
   ```

2. **📦 Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **📥 Download test data**
   - Hugging Face Dataset: [Download Link](https://huggingface.co/datasets/LivXue/SoMe)
   - Google Drive: [Download Link](https://drive.google.com/file/d/1sD2EaZStK5nODQWlJTHZ8WfFb5QHgwMN/view?usp=drive_link)  
   - Baidu Disk: [Download Link](https://pan.baidu.com/s/1DugTyLR5AaQHeOdXG6wqQQ?pwd=SoMe) (Password: SoMe)
   
   After downloading, unzip the data into the `database` directory.

---

## 💻 Usage

### 🏃‍♂️ Running Individual Tasks

Each task can be evaluated using its corresponding test script:

```bash
# 🚨 Realtime Event Detection
python test_realtime_event_detection.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 📊 Streaming Event Summary
python test_streaming_event_summary.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 🚫 Misinformation Detection
python test_misinformation_detection.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 🎯 User Behavior Prediction
python test_user_behavior_prediction.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 😊 User Emotion Analysis
python test_user_emotion_analysis.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 💬 User Comment Simulation
python test_user_comment_simulation.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# 📱 Media Content Recommendation
python test_media_content_recommend.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY

# ❓ Social Media Question Answering
python test_social_media_question_answering.py --model MODEL_NAME --base_url MODEL_SERVER_URL --api_key API_KEY
```

### ⚙️ Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | The model name to use | "deepseek-chat" |
| `--base_url` | The base URL for the model server | "https://api.deepseek.com" |
| `--api_key` | The API key for the model server | Your actual API key |
| `--output_path` | Output path for results | "results/my_experiment" |

### 📊 Evaluation

After running the test scripts, evaluate the results using the provided evaluation scripts:

```bash
# Option 1: For tasks with LLM-based answer extraction
python eval_scripts/[TASK]_extraction.py
python eval_scripts/[TASK]_compute_score.py

# Option 2: For tasks with LLM-as-judge scoring
python eval_scripts/[TASK]_scoring.py
python eval_scripts/[TASK]_compute_score.py
```

> **Note**: The LLM settings for evaluation are configured in `eval_scripts/settings.json`

---

## 🧠 Model Support

The benchmark supports various LLM models through OpenAI-compatible API endpoints:

- 🧩 **Qwen series models** (Qwen2.5, Qwen3, etc.)
- 🔌 **OpenAI models** (GPT-4, GPT-5, etc.)
- 🌐 **Third-party models** with OpenAI-compatible APIs (DeepSeek, Claude, etc.)
- 📦 **Local models** served with OpenAI-compatible wrappers (vLLM, Ollama, etc.)

---

## 📚 Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@inproceedings{some2026,
  title={SoMe: A Realistic Benchmark for LLM-based Social Media Agents},
  author={Dizhan Xue and Jing Cui and Shengsheng Qian and Chuanrui Hu and Changsheng Xu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 🤝 Contributing

We welcome contributions to improve the benchmark! Here's how you can help:

1. **🐛 Report bugs** by opening issues with detailed descriptions
2. **💡 Suggest features** for new tasks or improvements
3. **🔧 Submit code** via pull requests for bug fixes or enhancements
4. **📊 Add datasets** to expand the benchmark coverage
5. **📝 Improve documentation** for better usability

Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We would like to express our gratitude to:

- The **Qwen team** for their excellent Qwen-Agent framework, which forms the foundation of this benchmark
- All contributors who have helped develop and improve SoMe
- The social media platforms and data providers that make this research possible
- The AAAI 2026 reviewers for their valuable feedback

---

## ❓ Q&A

### Why did the author modify the Qwen-Agent repository?
Early agentic LLMs each had their own tool calling formats, and they could not properly follow prompts to use Qwen's calling format. Therefore, I added support for other tool calling formats in Qwen-Agent. If you are using Qwen models or relatively new agentic LLMs, you can directly use the original Qwen-Agent repository.

---

## 📞 Contact

For questions or inquiries about the benchmark, please contact:

- Dizhan Xue: xuedizhan17@mails.ucas.ac.cn

Visit our [GitHub repository](https://github.com/LivXue/SoMe) for the latest updates and discussions.
