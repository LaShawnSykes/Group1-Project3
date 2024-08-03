# Neural Newsroom with Guardian Article Classifier

## Project Overview
Neural Newsroom is an AI-powered news summarization and categorization tool that fetches recent articles from The Guardian, classifies them using a CNN-LSTM hybrid model, and provides concise summaries on user-specified topics. This project combines a robust backend for article classification with a user-friendly front-end interface.

## Key Features
1. Data Collection: Fetching articles from The Guardian API
2. Data Preprocessing: Cleaning and preparing text data for model input
3. Model Development: CNN-LSTM hybrid architecture for article classification
4. Model Evaluation: Assessing the model's performance
5. Prediction: Using the trained model to classify new articles
6. Summarization: Generating concise summaries for each news category
7. PDF Generation: Creating downloadable PDF reports
8. User Interface: Gradio-powered interface for easy interaction

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7+
- pip (Python package manager)
- A Guardian API key (get one [here](https://open-platform.theguardian.com/access/))

## Installation
1. Clone the repository:
git clone (https://github.com/LaShawnSykes/Neural-Newsroom_G2_P3.git)
cd neural-newsroom

2. Install the required packages:
pip install -r requirements.txt

3. Create a `.env` file in the project root and add your Guardian API key:
GUARDIAN_API_KEY=your_api_key_here

## Usage
1. Run the Jupyter notebook:
jupyter notebook

2. Open the `NeuralNewInterface.ipynb` notebook and run all cells.

3. The Gradio interface will launch, allowing you to enter a topic, choose language and sorting options, and specify the number of articles to fetch.

4. Click "Submit" to generate a news summary and PDF report.

## Detailed Steps
### 1. Data Collection
- Utilized The Guardian API to fetch articles
- Implemented rate limiting and error handling for robust data collection
- Stored fetched articles in a CSV file for further processing

### 2. Data Preprocessing
- Cleaned text data by removing special characters and converting to lowercase
- Tokenized the text and removed stop words
- Encoded article categories using LabelEncoder
- Padded sequences to ensure uniform input size

### 3. Model Development
We experimented with several model architectures, gradually refining our approach:
#### Initial CNN-LSTM Hybrid Model
- Embedding layer
- Conv1D layers with MaxPooling
- LSTM layer
- Dense layers with dropout for classification

#### Refined CNN-LSTM Hybrid Model
- Adjusted the architecture to reduce overfitting
- Implemented batch normalization
- Increased dropout rates
- Fine-tuned hyperparameters like learning rate and regularization

#### Further Refined CNN-LSTM Hybrid Model
- Simplified the architecture
- Adjusted learning rate schedule
- Increased L2 regularization
- This version showed the best balance between performance and generalization

### 4. Model Evaluation
- Used accuracy and loss metrics during training
- Implemented early stopping to prevent overfitting
- Evaluated the model using classification report and confusion matrix
- Visualized training history to understand model learning patterns

### 5. Prediction and Summarization
- Developed functions to predict categories for new articles and generate summaries
- Implemented text summarization techniques for each category
- Created a system to generate top article snippets

### 6. PDF Generation
- Implemented functionality to create downloadable PDF reports of the summaries

### 7. User Interface
- Developed a Gradio-powered interface for easy interaction with the system

## File Descriptions
- `NeuralNewsroomModel.ipynb`: Jupyter notebook containing the model code
- `NeuralNewsInterface.py`: Script for fetching articles from The Guardian API
- `preprocess.py`: Data preprocessing functions
- `model.py`: CNN-LSTM hybrid model architecture
- `train.py`: Script for training the model
- `evaluate.py`: Functions for model evaluation
- `predict.py`: Script for making predictions on new articles
- `label_encoder_final.pickle`: Pickle file containing the trained label encoder
- `requirements.txt`: List of Python dependencies
- `.env`: Environment file for storing the API key (not included in the repository)

## Model Information
The current version uses a CNN-LSTM hybrid model for article classification. While it provides good results, it may benefit from further training on a larger dataset for improved accuracy.

## Future Improvements
- Train the classification model on a larger, more diverse dataset
- Experiment with more advanced architectures (e.g., Transformer-based models)
- Implement cross-validation for more robust evaluation
- Explore transfer learning using pre-trained language models
- Implement more advanced NLP techniques for better summarization
- Add support for multiple news sources
- Improve error handling and API request management
- Develop a more sophisticated frontend

## Contributing
Contributions to Neural Newsroom are welcome. Please feel free to submit a Pull Request.

## Contributors
- La Shawn Sykes
- Frank Hanan

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Guardian for providing the news API
- Gradio for the easy-to-use interface framework
- TensorFlow and Keras for machine learning capabilities
- Claude for suggetions and code remediation guidance
