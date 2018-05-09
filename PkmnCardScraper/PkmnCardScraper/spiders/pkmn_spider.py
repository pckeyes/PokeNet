# -*- coding: utf-8 -*-
import scrapy


class PkmnSpiderSpider(scrapy.Spider):
    name = 'pkmn_spider'
    #allowed_domains = ['https://pkmncards.com/?s=e%3Abase-set&display=card&sort=number']
    start_urls = ['https://pkmncards.com/?s=e%3Abase-set&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ajungle&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Afossil&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Abase-set-2&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ateam-rocket&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Agym-heroes&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Agym-challenge&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aneo-genesis&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aneo-discovery&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aneo-revelation&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aneo-destiny&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aexpedition&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aaquapolis&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Askyridge&display=card&sort=number']

    def parse(self, response):
        titles = response.xpath('//span[@itemprop="name"]/text()').extract()
        #split the string to get name and set
        low_prices = response.xpath('//div[@class = "low"]/a/text()').extract()
        mid_prices = response.xpath('//div[@class = "mid"]/a/text()').extract()
        high_prices = response.xpath('//div[@class = "hi"]/a/text()').extract()
        imgs = response.css('.scan.left a img::attr(src)').extract()
        #pkmn_types = response.css('.text p:nth-child(1)').extract()
        pkmn_types = response.xpath('//div[@class = "text"]').extract()

 
        for item in zip(titles, low_prices, mid_prices, high_prices, imgs, pkmn_types):
            #creat dict
            scraped_data = {
                    "title": item[0],
                    "low price": item[1],
                    "mid price": item[2],
                    "high price": item[3],
                    "img": item[4],
                    "type": item[5]
            }

            #yeild data
            yield(scraped_data)