import re
from contextlib import closing
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd


class FlipkartReviewScraper:
    def __init__(self, website_url, max_pages=12):
        """
        Initialize the scraper with target URL and optional page limit
        
        Args:
            website_url (str): URL of the Flipkart product reviews page
            max_pages (int): Maximum number of pages to scrape (default: 12)
        """
        self.website_url = website_url
        self.max_pages = max_pages
        self.reviews_data = []
        
        # Configure Selenium options
        self.driver_options = webdriver.FirefoxOptions()
        self.driver_options.add_argument('--headless')  # Run in headless mode

    def remove_non_ascii(self, text):
        """Remove non-ASCII characters from text."""
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

    def extract_review_content(self, tag):
        """Extract and clean review content from BeautifulSoup tag."""
        content_div = tag.find("div", class_="qwjRop")
        if not content_div or not content_div.div:
            return None

        content = content_div.div.prettify()
        content = (content.replace(u"\u2018", "'")
                  .replace(u"\u2019", "'")
                  [24:-35])
        content = self.remove_non_ascii(content)
        return re.sub('<.+>', ' ', content)

    def navigate_to_page(self, browser, page_num):
        """Navigate to specific page number in pagination."""
        nav_btns = browser.find_elements(By.CLASS_NAME, '_2Xp0TH')
        button = next((btn for btn in nav_btns 
                      if btn.text.isdigit() and int(btn.text) == page_num), None)
        
        if not button:
            print(f"Page {page_num} button not found")
            return False
        
        button.send_keys(Keys.RETURN)
        WebDriverWait(browser, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "_2xg6Ul")))
        return True

    def expand_read_more_buttons(self, browser):
        """Click all 'Read More' buttons to expand full reviews."""
        read_more_btns = browser.find_elements(By.CLASS_NAME, '_1EPkIx')
        for btn in read_more_btns:
            browser.execute_script("arguments[0].scrollIntoView();", btn)
            browser.execute_script("window.scrollBy(0, -150);")
            btn.click()

    def scrape_page_reviews(self, browser):
        """Scrape all reviews from the current page."""
        soup = BeautifulSoup(browser.page_source, "lxml")
        return soup.find_all("div", class_="_1PBCrt")

    def process_review(self, review):
        """Process individual review and return extracted data."""
        try:
            title = review.find("p", class_="_2xg6Ul").string
            title = self.remove_non_ascii(title)
            
            content = self.extract_review_content(review)
            if not content:
                return None

            votes = review.find_all("span", class_="_1_BQL8")
            upvotes = int(votes[0].string) if len(votes) > 0 else 0
            downvotes = int(votes[1].string) if len(votes) > 1 else 0

            return title, content, upvotes, downvotes
        except Exception as e:
            print(f"Error processing review: {e}")
            return None

    def scrape_page(self, browser, page_num):
        """Scrape reviews from a single page."""
        if not self.navigate_to_page(browser, page_num):
            return []
            
        self.expand_read_more_buttons(browser)
        reviews = self.scrape_page_reviews(browser)
        
        page_data = []
        for review in reviews:
            review_data = self.process_review(review)
            if review_data:
                page_data.append(review_data)
                self.reviews_data.append([review_data[1]])  # Store just content for CSV
                
        return page_data

    def save_to_text_file(self, filename, data):
        """Save reviews to a text file."""
        with open(filename, "w", encoding="utf-8") as file:
            for title, content, upvotes, downvotes in data:
                file.write(f"Review Title: {title}\n\n")
                file.write(f"Upvotes: {upvotes}\nDownvotes: {downvotes}\n\n")
                file.write(f"Review Content:\n{content}\n\n\n")

    def save_to_csv(self, filename):
        """Save reviews to a CSV file."""
        df = pd.DataFrame(self.reviews_data, columns=['Review Content'])
        df.to_csv(filename, index=False)

    def run(self):
        """Execute the scraping process."""
        with closing(webdriver.Firefox(options=self.driver_options)) as browser:
            browser.get(self.website_url)
            
            all_reviews = []
            for page_num in range(1, self.max_pages + 1):
                print(f"Scraping page {page_num}...")
                page_reviews = self.scrape_page(browser, page_num)
                all_reviews.extend(page_reviews)

            # Save outputs
            self.save_to_text_file("Reviews.txt", all_reviews)
            self.save_to_csv("Review_Contents.csv")
            print("Scraping completed successfully!")


if __name__ == "__main__":
    # Example usage
    product_url = "https://www.flipkart.com/redmi-note-5-pro-black-64-gb/product-reviews/itmf2fc3xgmxnhpx?page=1&pid=MOBF28FTQPHUPX83"
    
    # Create scraper instance with URL
    scraper = FlipkartReviewScraper(website_url=product_url, max_pages=5)
    
    # Run the scraper
    scraper.run()