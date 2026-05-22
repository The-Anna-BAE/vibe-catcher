from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

from bs4 import BeautifulSoup
import json

import re
import asyncio
import sys
import numpy as np

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'



def get_links(name,url):
        if name in url:
            return url
        
        if sys.platform == 'win32':
            if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with Stealth().use_sync(sync_playwright()) as playwright:
                browser = playwright.chromium.launch(headless=False)
                context = browser.new_context(
                                                user_agent=user_agent,
                                                viewport={'width': 1920, 'height': 1080}, 
                                                device_scale_factor=1,
                                                is_mobile=False,
                                                has_touch=False,
                                                locale="ru-RU",
                                                timezone_id="Europe/Minsk"
                                            )
                page = browser.new_page()

                url= re.search(r'^(.*/maps/)',url)
                if url:
                    url=url.group(1)

                page.goto(url+'?text='+name)
                page.wait_for_selector('.search-list-view__list')
                page.mouse.move(201,513)
                for i in range(4):
                    page.mouse.wheel(0,3000)
                    page.wait_for_timeout(np.random.randint(1000,3001))

                stores_lists = page.locator('a[href*="/reviews/"]')

                sup_url= url.split('/maps')[0]
                links = stores_lists.all()
                unique_links = set()
                for link in links:
                    href= link.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            new_url= sup_url+href
                        else:
                            new_url=sup_url+'/'+href

                        unique_links.add(new_url)
                browser.close()
                return unique_links
        except Exception as error:
            print(f"Беда. Что-то не получается {error} (0(0(0(0 ")
            return set()



def scrapper(url,user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36', timezone_id ="Europe/Minsk" ):

    if sys.platform == 'win32':
        if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with Stealth().use_sync(sync_playwright()) as playwright:
                browser = playwright.chromium.launch(headless=False)
                context = browser.new_context(
                                                user_agent=user_agent,
                                                viewport={'width': 1920, 'height': 1080}, 
                                                device_scale_factor=1,
                                                is_mobile=False,
                                                has_touch=False,
                                                locale="ru-RU",
                                                timezone_id=timezone_id
                                            )
                page = browser.new_page()
                page.goto(url)
                
                # print("Жду загрузку...")
                page.wait_for_timeout(5000)
                page.mouse.move(354,557)
                
                for _ in range(40):
                    page.mouse.wheel(0,3000)
                    page.wait_for_timeout(np.random.randint(1000,1501))

            
                reviews = page.evaluate("""
                    () => {
                        const results = [];
                        
                        const elements = document.querySelectorAll('span, div');
                        
                        elements.forEach(el => {
                            const text = el.innerText ? el.innerText.trim() : "";

                            if (text.length > 30 && !text.includes('{') && el.children.length === 0) {
                                results.push({
                                    "brand": "brand_name",
                                    "text": text.replace(/\\n/g, ' ')
                                });
                            }
                        });
                        return results;
                    }
                """)
                
                browser.close()
                return reviews
        except Exception as e:
            print(f"Пупупу... {e}")



def put_reviews_to_file(data,path_to_file):

    stop_words = [
    "главная", "контакты", "каталог", "услуги", "новости", "faq",
    "поиск", "купить", "корзина", "заказать", "подписаться", "регистрация",
    "войти", "скачать", "подробнее", "отправить", "отмена",
    "имя", "телефон", "email", "пароль", "карта",
    "конфиденциальность", "профиль", "яндекс","погода", "вопросы", "оплата"
    ]


    with open(path_to_file, 'w', encoding='utf-8') as f:
        pass
    unique_data=[]

    for v in data:
        do_not=False
        if isinstance(v,dict):
            text = v.get('text','').lower()
        else:
            text = str(v).lower()
        for word in stop_words:
            if word in text:
                do_not=True
                break
        if not do_not and text not in unique_data:
            unique_data.append(text)
    if unique_data:
        with open(path_to_file, 'a', encoding='utf-8') as f:
            for review in unique_data:
                f.write(review.replace('\n','')+'\n')
        print(f"Собрано уникальных текстов: {len(unique_data)}")
    else:
        print("Пусто")