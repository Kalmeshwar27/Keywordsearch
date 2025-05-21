from flask import Flask, render_template, request, jsonify, send_file
import feedparser
import re
import nltk
from urllib.parse import urlparse
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from datetime import datetime, timedelta
import time
import csv
import io

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
blacklist = stop_words.union({
    "said", "will", "news", "india", "today", "report", "year", "week", "also", "time",
    "one", "two", "many", "more", "from", "about", "could", "back", "out", "into", "under",
    "over", "minister", "government", "officials", "party", "member", "states", "people",
    "country", "nation", "issue", "media", "world", "video", "audio", "language", "chilli",
    "pope", "explore", "powder", "every", "month", "daily", "newsroom", "click", "read",
    "update", "headline", "live"
})

rss_feeds = [
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://indianexpress.com/feed/",
    "https://www.financialexpress.com/feed/",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "https://www.livemint.com/rss/news",
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "https://www.france24.com/en/rss",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://moxie.foxnews.com/feedburner/latest.xml",
    "https://abcnews.go.com/abcnews/topstories",
    "https://feeds.skynews.com/feeds/rss/home.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.theguardian.com/world/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

now = datetime.utcnow()
time_limit = now - timedelta(days=15)

keyword_to_domains = defaultdict(set)
keyword_to_articles = defaultdict(list)

def fetch_and_process():
    global keyword_to_domains, keyword_to_articles
    keyword_to_domains.clear()
    keyword_to_articles.clear()

    for feed_url in rss_feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            published_struct = entry.get('published_parsed') or entry.get('updated_parsed')
            if not published_struct:
                continue
            published_datetime = datetime.fromtimestamp(time.mktime(published_struct))
            if published_datetime < time_limit:
                continue

            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            description = re.sub(r'<[^>]+>', '', entry.get("summary", "")).strip()
            domain = urlparse(link).netloc

            if not title or not link:
                continue

            combined_text = f"{title}. {description}"
            tokens = word_tokenize(combined_text)
            tagged_words = pos_tag(tokens)

            nouns = [
                word.lower() for word, tag in tagged_words
                if tag.startswith('NN') and word.isalpha() and len(word) > 3 and word.lower() not in blacklist
            ]
            top_nouns = set([word for word, _ in Counter(nouns).most_common(5)])

            for kw in top_nouns:
                keyword_to_domains[kw].add(domain)
                keyword_to_articles[kw].append({
                    "title": title,
                    "link": link,
                    "published_date": published_datetime.strftime("%Y-%m-%d"),
                    "published_time": published_datetime.strftime("%I:%M:%S %p")
                })

fetch_and_process()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    keyword = data.get("keyword", "").lower().strip()
    page = int(data.get("page", 1))
    per_page = 5

    if not keyword:
        return jsonify({"error": "Please enter a keyword."}), 400

    domains = keyword_to_domains.get(keyword)
    articles = keyword_to_articles.get(keyword)
    if not domains or not articles or len(domains) < 2:
        return jsonify({"error": f"No common articles found for '{keyword}'."}), 404

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    paged_articles = articles[start:end]

    return jsonify({
        "keyword": keyword,
        "domains": list(domains),
        "articles": paged_articles,
        "total_articles": len(articles),
        "page": page,
        "per_page": per_page
    })

@app.route("/download_csv", methods=["POST"])
def download_csv():
    data = request.json
    keyword = data.get("keyword", "").lower().strip()

    if not keyword:
        return jsonify({"error": "Please enter a keyword."}), 400

    domains = keyword_to_domains.get(keyword)
    articles = keyword_to_articles.get(keyword)
    if not domains or not articles or len(domains) < 2:
        return jsonify({"error": f"No common articles found for '{keyword}'."}), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Keyword", "Article Title", "Article Link", "Published Date", "Published Time"])

    for article in articles:
        writer.writerow([
            keyword,
            article["title"],
            article["link"],
            article["published_date"],
            article["published_time"]
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{keyword}_articles.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
