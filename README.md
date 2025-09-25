# 📄 PDF Chatbot with Llama 3, LangChain, and Pinecone

A powerful, end-to-end chatbot application that allows users to have an interactive conversation with their PDF documents. The app leverages a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based solely on the content of the uploaded PDF.

![Application Screenshot](https://via.placeholder.com/800x400?text=Chatbot+Interface+Screenshot)

## ✨ Features

- **Interactive UI**: User-friendly and responsive web interface built with Streamlit
- **Dynamic PDF Upload**: Upload any PDF directly through the application for processing
- **Local LLM**: Utilizes the powerful Llama 3 model running locally via Ollama for enhanced privacy and control
- **Vector Database**: Employs Pinecone's serverless vector database for efficient and scalable document chunk storage and retrieval
- **Source Verification**: Displays the exact text chunks from the PDF that were used to generate answers, ensuring transparency and trust
- **Modular Code**: Backend logic (`rag_logic.py`) is neatly separated from frontend UI (`app.py`) for better maintainability and scalability

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **Framework**: LangChain, Streamlit
- **LLM**: Llama 3 (via Ollama)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Pinecone
- **Core Libraries**: PyPDF, python-dotenv

## 🏗️ System Architecture

The application follows a Retrieval-Augmented Generation (RAG) workflow to ensure that answers are grounded in the content of the user's document:

1. **Upload & Process**: PDF is uploaded and split into smaller text chunks, converted into numerical vectors (embeddings)
2. **Store**: Embeddings are stored in a Pinecone serverless vector index for fast retrieval
3. **Query**: User questions are converted into embeddings
4. **Retrieve**: System queries Pinecone to find the most semantically relevant text chunks
5. **Generate**: Retrieved chunks and user questions are passed to Llama 3 model for answer generation

![RAG Architecture Diagram](https://via.placeholder.com/600x300?text=RAG+System+Architecture+Diagram)

## 📋 Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Python 3.8+**
- **Ollama**: Download and install from [ollama.com](https://ollama.com)
- **Llama 3 Model**: Pull the model by running:
  ```bash
  ollama pull llama3
  ```
- **Pinecone Account**: 
  - Sign up for a free account at [pinecone.io](https://pinecone.io)
  - Get your API Key from the dashboard

## 🚀 Setup & Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd pdf-chatbot-llama3
```

### Step 2: Create a Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Environment File

Create a `.env` file in the root directory and add your Pinecone API Key:

```env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY_HERE"
```

## ▶️ Usage

### Step 1: Ensure Ollama is Running
Make sure the Ollama application or server is running in the background.

### Step 2: Launch the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Use the Chatbot
1. Your web browser should open with the application
2. In the sidebar, upload your PDF and click "Process Document"
3. Wait for the confirmation message
4. Start asking questions in the main chat window

![PDF Upload Interface](https://via.placeholder.com/400x300?text=PDF+Upload+Sidebar)

## 📂 File Structure

```
pdf-chatbot-llama3/
├── app.py              # Main Streamlit application file
├── rag_logic.py        # Backend RAG logic module
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

### File Descriptions

- **`app.py`**: Main Streamlit application handling UI, state management, and backend calls
- **`rag_logic.py`**: Core RAG logic for PDF processing, embedding, and Pinecone storage
- **`requirements.txt`**: List of all required Python packages
- **`.env`**: Secure storage for API keys and environment variables

## 🧰 Troubleshooting

### Common Issues

**Pinecone Index Creation Error**
```
Pinecone.create_index() missing 1 required positional argument: 'spec'
```
*Solution*: The provided `rag_logic.py` file has been updated to handle the latest Pinecone client library.

**Connection Errors**
- Ensure Ollama is running before starting the Streamlit app
- Verify your `PINECONE_API_KEY` in the `.env` file is correct
- Check your internet connection for Pinecone API access

**Model Loading Issues**
- Confirm Llama 3 model is downloaded: `ollama list`
- Restart Ollama service if needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) for local LLM deployment
- [Pinecone](https://pinecone.io) for vector database services
- [LangChain](https://langchain.com) for RAG framework
- [Streamlit](https://streamlit.io) for the web interface

## 📧 Contact

**Created by Tushar Saxena**

Project Link: [https://github.com/yourusername/pdf-chatbot-llama3](https://github.com/yourusername/pdf-chatbot-llama3)

---

⭐ If you found this project helpful, please give it a star!
