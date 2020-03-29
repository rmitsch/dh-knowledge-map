# DH Education Knowledge Map - creating knowledge maps via hypertext

About this project: [DH Education Knowledge Map - creating knowledge maps via hypertext](www.dontknowtheurlyet.com)


### Setup Instructions

#### With conda

Change to source directory.  
Create environment with:
```bash
conda env create --name dhekm --file=environments.yml
```

Run with: 
```bash
conda activate dhekm
python exploration.py
```

#### With Docker 

Run w/o building by downloading from Docker Hub with: 
```bash
docker run -p 8050:8050 rmitsch/dh-knowledge-map python exploration.py
```

Build with:
```bash
docker build -t dh-knowledge-map -f Dockerfile .
```

Run after local build with: 
```bash
docker run -p 8050:8050 dh-knowledge-map python exploration.py
```
