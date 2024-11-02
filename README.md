
# From NBE Bank project
# Social Media Responder: AI-Driven Customer Interaction Solution

## Project Overview

The **Social Media Responder** project leverages advanced AI models to enhance customer satisfaction through automated, accurate, and prompt responses across social media platforms. The solution aims to reduce the strain of high inquiry volumes on customer support teams while addressing user expectations for fast replies, ultimately fostering a positive customer experience.

### Objectives
- Improve customer interactions on social media.
- Implement AI-powered automation for real-time responses.
- Classify customer feedback to assess satisfaction levels.

## Solution Components

### 1. Generative AI with Retrieval-Augmented Generation (RAG)
Our solution is built using the following technologies:
   - **Gen-AI Models**: ChatGroq and Llama 3.1 models for high-speed, multilingual processing.
   - **Retrieval-Augmented Generation (RAG)**: Enables efficient document embedding, retrieval, and generation, utilizing FAISS as a vector database for quick and accurate responses.
   - **Linguistic Processing Units (LPU)**: Powered by Groq’s architecture, achieving approximately 500 tokens per second.

### 2. Customer Feedback Classification
The project also includes a customer feedback analysis pipeline to gauge satisfaction, supporting long-term ROI:
   - **Data Collection**: App Store and Play Store reviews (Arabic only).
   - **Data Preprocessing**: Tokenization, stopword removal, and lemmatization using NLTK.
   - **Model Training**: Classification using a fine-tuned RoBERTa model, assessing sentiments as positive, negative, or neutral.

## Key Strengths

- **High-Speed Processing**: Groq chips provide rapid response generation, ensuring seamless customer interactions.
- **Llama 3.1 Model**: This LLM supports an extensive context window (up to 128,000 tokens) and offers multilingual capabilities and enhanced safety.
- **Enhanced Retrieval with RAG**: The retrieval chain setup improves response relevance by accurately matching queries with stored customer support knowledge.

## Deployment

The deployment of this AI-driven solution enables companies to engage with customers promptly on social media, fostering loyalty and customer satisfaction.


## References

For more details on the components used in this project, please refer to:
- [Microsoft on social media as a customer service channel](https://www.microsoft.com/en-us/dynamics-365/blog/business-leader/2016/09/02/64-of-millennials-say-social-media-is-an-effective-customer-service-channel/)
- [Forrester insights on customer expectations](https://www.forrester.com/blogs/16-03-03-your_customers_dont_want_to_call_you_for_support/)
- [Statista statistics on social media response expectations](https://www.statista.com/statistics/808477/expected-response-time-for-social-media-questions-or-complaints/)
