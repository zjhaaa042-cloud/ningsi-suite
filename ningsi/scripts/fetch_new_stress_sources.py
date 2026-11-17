"""Inspect public Mendeley stress archives and fetch selected members with CRC verification."""
import argparse
import json
import struct
import zipfile
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests


def mendeley_index(dataset, root):
    directory = root/dataset
    directory.mkdir(exist_ok=True,parents=True)
    info = requests.get(f'https://data.mendeley.com/public-api/datasets/{dataset}/files',params={'folder_id':'root','version':'1'},timeout=40)
    info.raise_for_status()
    item, = info.json()
    details = item['content_details']
    url = details['download_url']
    size = details['size']
    start = max(0,size-1024*1024)
    data = byte_range(url,start,size-1)
    # Standard library handles ZIP64 and central-directory member offsets on a sparse virtual file.
    class Tail:
        pos = size
        def tell(self): return self.pos
        def seek(self, offset, whence=0):
            self.pos = (0,self.pos,size)[whence]+offset
            return self.pos
        def read(self, n=-1):
            if self.pos < start: raise ValueError('Archive index exceeds expected tail')
            chunk = data[self.pos-start:] if n<0 else data[self.pos-start:self.pos-start+n]
            self.pos += len(chunk)
            return chunk
    with zipfile.ZipFile(Tail()) as archive:
        entries=[dict(name=i.filename,length=i.file_size,compressed=i.compress_size,offset=i.header_offset,crc32=i.CRC,method=i.compress_type) for i in archive.infolist() if not i.is_dir()]
    index=dict(url=url,size=size,full_sha256=details['sha256_hash'],files=entries)
    (directory/'index.json').write_text(json.dumps(index,indent=2),encoding='utf8')
    return index


def byte_range(url, start, stop):
    r=requests.get(url,headers={'Range':f'bytes={start}-{stop}'},stream=True,timeout=(30,120))
    with r:
        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{stop}/'):
            raise ValueError(f'Bad range response {r.status_code} {r.headers.get("Content-Range")}')
        data=r.content
    if len(data)!=stop-start+1: raise ValueError('Truncated range')
    return data


def member_bytes(index, entry):
    start=entry['offset']
    # Local ZIP headers occasionally contain extra data; a single ranged transfer is typical.
    end=min(index['size']-1,start+entry['compressed']+4095)
    data=byte_range(index['url'],start,end)
    if data[:4] != b'PK\x03\x04': raise ValueError('Missing local ZIP header')
    name_n,extra_n=struct.unpack('<HH',data[26:30])
    h=30+name_n+extra_n
    comp=data[h:h+entry['compressed']]
    if len(comp)!=entry['compressed']:
        comp=byte_range(index['url'],start+h,start+h+entry['compressed']-1)
    decoded=zlib.decompress(comp,-15) if entry['method']==8 else comp
    if entry['method'] not in (8,0) or len(decoded)!=entry['length'] or zlib.crc32(decoded)!=entry['crc32']:
        raise ValueError('Corrupt member '+entry['name'])
    return decoded


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('datasets',nargs='+')
    opts=p.parse_args()
    def read(d):
        obj=mendeley_index(d,opts.root)
        print(d,len(obj['files']),[(e['name'],e['length']) for e in obj['files'][:8]])
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(read,opts.datasets))
