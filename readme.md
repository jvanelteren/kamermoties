## Kamermoties
Project to explore Dutch parliament resolutions since 2007. Published at [blog](https://jvanelteren.github.io/blog/)

### Usage
The core workflow consists of a couple of jupyter notebooks. In between pickle files are generated
* 1_get_data ➡ downloads data from Tweede Kamer API, preprocesses
* 2_2021-02-20-kamermotiesEDA ➡ exploratory data analysis
* 3_top2vec_analysis ➡ clusters resolutions into topics, prepares slimmed down file for production
* 4_2021-03-07-kamermoties_topics ➡ visualizes results of topics
So notebook 2 & 4 are visualisations


App.py is an app I've made to interactively explore the topics

### Thanks
Thanks to [https://opendata.tweedekamer.nl/](https://opendata.tweedekamer.nl/) for making the API available

### License
[![License: Creative Commons Naamsvermelding-GelijkDelen 4.0 Internationaal-licentie](https://i.creativecommons.org/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/3.0/) 