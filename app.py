import gradio as gr
from news_fetcher import get_news_summary
from guardian_model import predict_article_category, preprocess_text
from tensorflow.keras.models import load_model
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Load the model and necessary components
model = load_model('guardian_article_classifier_final.h5')
with open('tokenizer_final.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder_final.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

max_len = 200  # Make sure this matches the value used during training

def format_output(output):
    html = f"""
    <style>
    .newspaper {{
        font-family: 'Times New Roman', Times, serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f9f7f1;
        border: 1px solid #d3d3d3;
    }}
    .newspaper h1 {{
        font-size: 36px;
        text-align: center;
        border-bottom: 2px solid #000;
        padding-bottom: 10px;
    }}
    .newspaper h2 {{
        font-size: 24px;
        margin-top: 20px;
    }}
    .newspaper p {{
        font-size: 16px;
        line-height: 1.6;
        text-align: justify;
    }}
    </style>
    <div class="newspaper">
        <h1>News Summary</h1>
        {output.replace('\n', '<br>')}
    </div>
    """
    return html

def generate_pdf(content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    story.append(Paragraph("News Summary", styles['Title']))
    story.append(Spacer(1, 12))

    # Add content
    for line in content.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def get_news_summary_with_pdf(topic, language='en', sort='published_desc', limit=10):
    articles = get_news_summary(topic, language, sort, limit)
    
    # Classify articles
    classified_articles = []
    for article in articles:
        title = article.get('title', '')
        body = article.get('content', '') or article.get('description', '')
        category = predict_article_category(title, body, model, tokenizer, label_encoder)
        article['category'] = category
        classified_articles.append(article)

    # Group articles by category
    categorized_articles = {}
    for article in classified_articles:
        category = article['category']
        if category not in categorized_articles:
            categorized_articles[category] = []
        categorized_articles[category].append(article)

    # Generate summary for each category
    output = f"Summary of recent news on '{topic}':\n\n"
    for category, cat_articles in categorized_articles.items():
        output += f"{category.upper()}:\n"
        cat_content = " ".join([art.get('content', '') or art.get('description', '') for art in cat_articles])
        cat_preprocessed = preprocess_text(cat_content)
        cat_summary = summarize_text(cat_preprocessed, num_sentences=2)
        output += f"{cat_summary}\n\n"

    # Add top snippets and sources
    top_snippets = get_top_snippets(articles)
    output += "Top Articles:\n"
    for i, snippet in enumerate(top_snippets, 1):
        output += f"{i}. {snippet}\n"

    sources = set(article.get('source', {}).get('name', 'Unknown') for article in articles)
    output += f"\nSources: {', '.join(sources)}"

    html_output = format_output(output)
    pdf_buffer = generate_pdf(output)
    return html_output, pdf_buffer

iface = gr.Interface(
    fn=get_news_summary_with_pdf,
    inputs=[
        gr.Textbox(label="Enter the topic you want a summary for:"),
        gr.Dropdown(choices=["en", "de", "es", "fr", "it", "nl", "no", "pt", "ru", "se", "zh"], label="Language", value="en"),
        gr.Dropdown(choices=["published_desc", "published_asc", "popularity"], label="Sort By", value="published_desc"),
        gr.Slider(minimum=1, maximum=25, step=1, label="Number of Articles", value=10)
    ],
    outputs=[
        gr.HTML(label="Summary and Sources"),
        gr.File(label="Download PDF")
    ],
    title="Neural Newsroom",
    description="Get a summary of the latest news on a given topic from multiple sources, presented in a newspaper-like format. You can also download the summary as a PDF.",
    examples=[["climate change"], ["artificial intelligence"], ["global economy"]],
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
