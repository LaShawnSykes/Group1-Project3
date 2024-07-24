![News](https://s36667.pcdn.co/wp-content/uploads/2020/12/News-cycle-GIF-747x400-B.gif)
![Good News](https://media.tenor.com/0oMWmMLQ9zMAAAAC/good-news.gif)

# Group1-Project3

# News Summarizer

## Problem Statement

In the contemporary digital landscape, users encounter a significant challenge in efficiently processing and synthesizing news information due to the exponential growth of online content sources. This information overload phenomenon presents a multifaceted problem: it impedes timely consumption of relevant data, obstructs the extraction of salient points from diverse sources, and complicates the formation of a coherent understanding of current events. The heterogeneity of news platforms, coupled with varying degrees of journalistic rigor and potential source biases, further exacerbates the complexity of information assimilation. Additionally, the absence of robust customization algorithms in many news aggregation systems limits users' capacity to filter and prioritize content according to their specific informational requirements.

Key challenges include: (Update with more info)

- Information Overload: Users struggle to process the vast volume of news content, leading to inefficient data consumption and potential oversight of critical information.

- Source Diversity and Quality Variance: The multiplicity of news sources with varying credibility and potential biases complicates the extraction of accurate, comprehensive information.

- Customization Limitations: Existing systems often lack robust algorithms for personalizing news feeds according to individual user preferences and requirements.

- Time and Language Constraints: Users face difficulties in rapidly assimilating pertinent information across linguistic boundaries within limited time frames.

## Solution (update at the very end)

The News Summarizer project addresses these challenges through a multi-faceted approach, leveraging advanced technologies and user-centric design. The solution effectively mitigates the issues of information overload, source diversity, customization limitations, and time and language constraints.

Key features implemented:(needs update)

- Multi-Source Aggregation and Summarization: The project utilizes APIs from NewsAPI and MediaStack to aggregate news from diverse sources, then employs Natural Language Processing (NLP) techniques to generate concise summaries. This approach directly tackles information overload by distilling large volumes of news into digestible content.

- Customizable User Interface: Through the implementation of a Gradio interface, users can specify topics of interest, preferred languages, sorting methods, and the number of articles to process. This customization addresses the limitation of personalization in existing systems, allowing users to tailor their news consumption experience.

- Language Support and Time Efficiency: The system supports multiple languages and provides real-time summarization, enabling users to quickly access relevant news across linguistic boundaries. This feature significantly reduces the time required to assimilate information from various sources.

## Features

- Aggregates news from multiple sources (NewsAPI and MediaStack)
- Generates concise summaries using NLP techniques
- Supports multiple languages
- Allows customization of news topics, language, and sorting preferences
- Provides real-time summarization
- Implements caching for improved performance

## Installation

1. Clone this repository:
   git clone <input the link to our github project>
cd news-summarizer

2. Install the required packages:
pip install requests gradio python-dotenv nltk beautifulsoup4

3. Set up your environment variables:
Create a `.env` file in the project root and add your API keys:
NEWSAPI_KEY=your_newsapi_key_here
MEDIASTACK_KEY=your_mediastack_key_here

## Usage

1. Run the Jupyter notebook:
  jupyter notebook

2. Open the `News_Summarizer.ipynb`<update this to the correct file name> file.

3. Run all cells in the notebook.

4. The Gradio interface will launch, allowing you to enter a topic, select a language, choose sorting options, and specify the number of articles to summarize.

5. Click "Submit" to get your news summary and top article snippets.

## Project Structure

- `News_Summarizer.ipynb`: Main Jupyter notebook containing all the code
- `news_cache/`: Directory for storing cached news data
- `.env`: File for storing API keys (not included in repository)

## Dependencies

- requests
- gradio
- python-dotenv
- nltk
- beautifulsoup4

## APIs Used

- [NewsAPI](https://newsapi.org/)
- [MediaStack](https://mediastack.com/)

## Limitations

- Uses free tier of NewsAPI and MediaStack, which have usage limits
- Summarization is based on a simple extractive method and may not always produce ideal results

## Future Improvements

- Implement more advanced summarization techniques
- Add support for more news sources
- Improve error handling and user feedback
- Implement a web-based frontend for easier access

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](<insert our github link>) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

Open the `News_Summarizer.ipynb` file and run all cells.

## Dependencies

- requests
- gradio
- python-dotenv
- nltk
- beautifulsoup4

## Acknowledgements

- [NewsAPI](https://newsapi.org/) for providing access to diverse news sources
- [MediaStack](https://mediastack.com/) for their comprehensive news data API
- [NLTK](https://www.nltk.org/) for natural language processing capabilities
- [Gradio](https://www.gradio.app/) for the user-friendly interface framework
- [Bing News Search API](https://www.microsoft.com/en-us/bing/apis) Offers search capabilities for news articles, enriched with rich metadata.
- [New York Times API](https://developer.nytimes.com/apis) Provides a comprehensive suite of news articles, book reviews, and bestseller lists from The New York Times.
- [GDELT Project](https://blog.gdeltproject.org/announcing-the-gdelt-context-2-0-api/) Monitors the world's news media, translating it nearly in real-time into     structured data.
- [Currents API](https://currentsapi.services/en/docs/) Provides the latest news published in various blogs, websites, and news outlets.
- [Event Registry](https://eventregistry.org/) Analyzes news from across the world in real-time, providing comprehensive insights.
- [Reuters News API](https://www.reuters.com/news-api/) Delivers trusted, timely, and comprehensive news from around the world.
- [MediaStack API](https://mediastack.com/) Access live news and historical articles from a wide range of sources with a fast and simple API.
- [ContextualWeb News API](https://rapidapi.com/contextualwebsearch/api/websearch)) Fetch timely news articles from thousands of sources globally.


