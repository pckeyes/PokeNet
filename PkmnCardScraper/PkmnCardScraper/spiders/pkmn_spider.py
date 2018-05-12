# -*- coding: utf-8 -*-
import scrapy


class PkmnSpiderSpider(scrapy.Spider):
    name = 'pkmn_spider'
    #allowed_domains = ['https://pkmncards.com/?s=e%3Abase-set&display=card&sort=number']
    #note left out World Collection, Victory Medals Sets b/c no prices
    #some links here have no prices are are breaking the json. need to check. Through base set 2 is good
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
                  'https://pkmncards.com/?s=e%3Askyridge&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aruby-sapphire&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asandstorm&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adragon&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ateam-magma-vs-team-aqua&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ahidden-legends&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Afirered-leafgreen&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ateam-rocket-returns&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adeoxys&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aemerald&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aunseen-forces&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adelta-species&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Alegend-maker&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aholon-phantoms&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Acrystal-guardians&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adragon-frontiers&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apower-keepers&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adiamond-pearl-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adiamond-pearl&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Amysterious-treasures&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asecret-wonders&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Agreat-encounters&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Amajestic-dawn&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Alegends-awakened&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Astormfront&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aplatinum&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Arising-rivals&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asupreme-victors&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aarceus&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aheartgold-soulsilver-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aheartgold-soulsilver&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aunleashed&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aundaunted&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Atriumphant&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Acall-of-legends&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Amcdonalds-collection-2012&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Amcdonalds-collection-2011&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ablack-white-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ablack-white&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aemerging-powers&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Anoble-victories&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adark-explorers&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adragons-exalted&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adragon-vault&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aboundaries-crossed&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aplasma-storm&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aplasma-freeze&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aplasma-blast&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Alegendary-treasures&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Axy-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Akalos-starter-set&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Axy&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aflashfire&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Afurious-fists&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aphantom-forces&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aprimal-clash&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Adouble-crisis&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aroaring-skies&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aancient-origins&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Abreakthrough&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Abreakpoint&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Agenerations&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Afates-collide&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asteam-siege&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aevolutions&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asun-moon-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asun-moon&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aguardians-rising&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aburning-shadows&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Ashining-legends&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Acrimson-invasion&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aultra-prism&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Aforbidden-light&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Awizards-black-star-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Asouthern-islands&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Alegendary-collection&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Abest-of-game&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Anintendo-black-star-promos&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Arumble&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-9&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-8&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-7&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-6&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-5&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-4&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-3&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-2&display=card&sort=number',
                  'https://pkmncards.com/?s=e%3Apop-series-1&display=card&sort=number']

    def parse(self, response):
        titles = response.xpath('//span[@itemprop="name"]/text()').extract()
        #split the string to get name and set
        low_prices = response.xpath('//div[@class = "low"]/a/text()').extract()
        mid_prices = response.xpath('//div[@class = "mid"]/a/text()').extract()
        high_prices = response.xpath('//div[@class = "hi"]/a/text()').extract()
        imgs = response.css('.scan.left a::attr(href)').extract()
        #imgs = response.css('.scan.left a img::attr(src)').extract()
        #pkmn_types = response.css('.text p:nth-child(1)').extract()
        pkmn_types = response.xpath('//div[@class = "text"]').extract()

 
        for item in zip(titles, low_prices, mid_prices, high_prices, imgs, pkmn_types):
            #creat dict
            scraped_data = {
                    "name": item[0],
                    "low price": item[1],
                    "mid price": item[2],
                    "high price": item[3],
                    "img": item[4],
                    "type": item[5]
            }

            #yeild data
            yield(scraped_data)