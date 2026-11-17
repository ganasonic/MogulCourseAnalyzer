"""Download only the pinned, licensed CPU inference model, with SHA256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def install():
    manifest=json.loads((ROOT/'models/yolox_manifest.json').read_text(encoding='utf-8'))
    path=ROOT/manifest['path']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']:
        print('Model already verified:',path)
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.onnx.part')
    try:
        digest=hashlib.sha256()
        size=0
        request=urllib.request.Request(manifest['url'],headers={'User-Agent':'MogulCourseAnalyzer'})
        with urllib.request.urlopen(request,timeout=45) as response,partial.open('wb') as stream:
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>manifest['size']:
                    raise ValueError('Unexpected model size')
                digest.update(chunk)
                stream.write(chunk)
        if size!=manifest['size'] or digest.hexdigest()!=manifest['sha256']:
            raise ValueError('Model checksum mismatch')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    print('Verified model:',path)
    return path


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    install()
