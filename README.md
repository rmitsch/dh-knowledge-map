# DH Education Knowledge Map - creating knowledge maps via hypertext

About this project: [DH Education Knowledge Map - creating knowledge maps via hypertext](https://medium.com/@marta.p/dh-education-knowledge-map-creating-knowledge-webs-via-hypertext-cfb6cc094c17).


### Setup Instructions

#### With conda

Change to source directory. Create environment with:
```bash
conda env create --name dhekm --file=environment.yml
```

Run with: 
```bash
conda activate dhekm
python exploration.py
```

Open [http://0.0.0.0:8050](http://0.0.0.0:8050).

#### With Docker 

_Downloading pre-built image from Docker hub_:
```bash
docker run -p 8050:8050 rmitsch/dh-knowledge-map python exploration.py
```
Open [http://0.0.0.0:8050](http://0.0.0.0:8050).

_Building the image locally_:
```bash
docker build -t dh-knowledge-map -f Dockerfile .
docker run -p 8050:8050 dh-knowledge-map python exploration.py
```
Open [http://0.0.0.0:8050](http://0.0.0.0:8050).

