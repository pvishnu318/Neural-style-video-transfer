import urllib.request
u="https://huggingface.co/datasets/Artificio/WikiArt/resolve/main/data/train-00000-of-00004-3c65976b59bc0ab4.parquet"
r=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"python"}),timeout=120)
tot=int(r.headers.get("Content-Length",0)); got=0
with open("data/wikiart_shard0.parquet","wb") as f:
    while True:
        b=r.read(1<<20)
        if not b: break
        f.write(b); got+=len(b)
print("done",got,tot)
