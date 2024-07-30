
def guardianapi():
    api_key = os.getenv('GUARDIAN_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1200)  # Fetch up to 1200 days of data
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    base_url = "https://content.guardianapis.com/search"
    
    sections = ['politics', 'business', 'technology', 'sport', 'culture', 'environment', 'science', 'world']
    all_articles = []

    def fetch_with_retry(params, max_retries=10, initial_wait=10):
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = min(initial_wait * (2 ** attempt), 60) + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Skipping this request.")
                    return None
                wait_time = min(initial_wait * (2 ** attempt), 60) + random.uniform(0, 1)
                print(f"Waiting for {wait_time:.2f} seconds before retrying.")
                time.sleep(wait_time)

    for section in sections:
        page = 1
        section_articles = []
        while len(section_articles) < 2000:
            params = {
                "api-key": api_key,
                "from-date": from_date,
                "to-date": to_date,
                "order-by": "relevance",
                "show-fields": "all",
                "page-size": 50,
                "section": section,
                "page": page
            }

            data = fetch_with_retry(params)
            if data is None:
                print(f"Failed to fetch articles from {section}, page {page}")
                break

            articles = data['response']['results']
            if not articles:
                break
            for article in articles:
                article['category'] = section
            section_articles.extend(articles)
            print(f"Fetched page {page} from {section}, total articles: {len(section_articles)}")
            page += 1

            # Random delay between requests
            time.sleep(random.uniform(1, 3))

        all_articles.extend(section_articles[:2000])  # Ensure we only take 2000 articles per section
        print(f"Completed fetching articles for {section}. Total articles: {len(all_articles)}")

    df = pd.DataFrame(all_articles)
    df.to_csv('.\\resources\\guardian_articles_cleaned.csv', index=False)
    print(f"Total articles fetched: {len(all_articles)}")
    print("DataFrame saved to 'guardian_articles_cleaned.csv'")
    print("Columns in the saved CSV:
          