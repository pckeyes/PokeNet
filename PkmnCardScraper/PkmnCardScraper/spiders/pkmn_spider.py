# -*- coding: utf-8 -*-
import scrapy


class PkmnSpiderSpider(scrapy.Spider):
    name = 'pkmn_spider'
    allowed_domains = ['https://pkmncards.com/?s=e%3Abase-set&display=card&sort=number']
    start_urls = ['https://pkmncards.com/?s=e%3Abase-set&display=card&sort=number']

    def parse(self, response):
        titles = response.xpath('//span[@itemprop="name"]/text()').extract()
        #split the string to get name and set
        low_prices = response.xpath('//div[@class = "low"]/a/text()').extract()
        mid_prices = response.xpath('//div[@class = "mid"]/a/text()').extract()
        high_prices = response.xpath('//div[@class = "hi"]/a/text()').extract()
        imgs = response.css('.scan.left a img::attr(src)').extract()
        pkmn_type = response.css('.text p:nth-child(1)::text').extract()

        for item in zip(titles, low_prices, mid_prices, high_prices, imgs, pkmn_type):
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