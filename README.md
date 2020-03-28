# dh-knowledge-map

### Setup Instructions

Run w/o building with: 
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
