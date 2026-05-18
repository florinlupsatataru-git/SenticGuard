import psycopg2
import logging
import toml
from newspaper import Article, Config
import newspaper
from transformers import pipeline

# Configure logging to monitor background execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_params():
    """Reads database credentials directly from the server's local secrets file."""
    try:
        # Streamlit uses this file on the server where the correct password is stored
        secrets = toml.load("/app/.streamlit/secrets.toml")
        return {
            "host": secrets["postgres"]["host"],
            "database": secrets["postgres"]["database"],
            "user": secrets["postgres"]["user"],
            "password": secrets["postgres"]["password"],
            "port": int(secrets["postgres"]["port"])
        }
    except Exception as e:
        logging.error(f"Error reading secrets.toml file: {e}")
        return None

def get_sources(cur):
    """Fetches the list of active news websites from the database."""
    cur.execute("SELECT url, site_name FROM news_sources;")
    return cur.fetchall()

def run_crawler():
    MAX_ARTICLES_PER_RUN = 50
    articles_saved = 0

    db_params = get_db_params()
    if not db_params:
        logging.error("Could not load database parameters. Stopping crawler.")
        return

    logging.info("Initializing AI pipeline (NewsAnalyzer)...")
    model_path = "florin-lupsa/NewsAnalyzer"
    try:
        classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        logging.error(f"Error loading AI model: {e}")
        return

    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config.memoize_articles = True # Avoid re-downloading previously processed articles

    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        return

    sources = get_sources(cur)
    logging.info(f"Found {len(sources)} sources to scan.")

    for source_url, site_name in sources:
        if articles_saved >= MAX_ARTICLES_PER_RUN:
            break

        logging.info(f"Scanning homepage: {site_name} ({source_url})...")
        try:
            paper = newspaper.build(source_url, config=config, keep_article_html=False)
        except Exception as e:
            logging.error(f"Error scanning source {site_name}: {e}")
            continue

        for article in paper.articles:
            if articles_saved >= MAX_ARTICLES_PER_RUN:
                break

            try:
                article.download()
                article.parse()

                title = article.title
                url = article.url
                content = article.text

                if not title or len(title) < 10 or not content:
                    continue

                # Initial AI model prediction based on the title
                prediction = classifier(title.strip()[:512])[0]
                label = prediction['label']
                score = float(prediction['score'])

                # Insert into the local pending queue database table
                cur.execute("""
                    INSERT INTO pending_queue (title, url, content, predicted_label, confidence_score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING;
                """, (title, url, content, label, score))

                if cur.rowcount > 0:
                    conn.commit()
                    articles_saved += 1
                    logging.info(f"[{articles_saved}/{MAX_ARTICLES_PER_RUN}] Successfully saved: {title[:40]}...")

            except Exception as e:
                continue

    logging.info(f"Crawling session finished. New articles added: {articles_saved}.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    run_crawler()
