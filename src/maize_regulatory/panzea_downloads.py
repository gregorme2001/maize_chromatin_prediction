"""Public Box discovery and checksum-verified resumable Panzea downloads."""

import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

SHARE = '9edkm5f51q88ltzrb0hxaz4p36fdsr7q'
FOLDER = f'https://iastate.box.com/s/{SHARE}/folder/364949363629'


def parse_box_listing(page):
    marker = 'Box.postStreamData = '
    if marker not in page:
        raise ValueError('Public Box folder data not found')
    data = json.JSONDecoder().raw_decode(page.split(marker, 1)[1])[0]
    entries = {}

    def visit(value):
        if isinstance(value, dict):
            name = str(value.get('name', ''))
            match = re.fullmatch(r'panzea_hapmap_282_chr(10|[1-9])\.vcf\.gz(\.tbi)?', name)
            if match:
                item = dict(chromosome=int(match[1]), name=name,
                            file_id=str(value['id']), bytes=int(value['itemSize']))
                if item['bytes'] <= 0 or (name in entries and entries[name] != item):
                    raise ValueError('Invalid/conflicting source metadata')
                entries[name] = item
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(data)
    expected = {f'panzea_hapmap_282_chr{c}.vcf.gz{suffix}'
                for c in range(1, 11) for suffix in ['', '.tbi']}
    if set(entries) != expected:
        raise ValueError('Expected all ten VCFs and their ten indexes')
    return sorted(entries.values(), key=lambda item: (item['chromosome'], item['name']))


def box_sha1(page):
    # The preview embeds JSON inside an escaped JavaScript string.
    values = set(re.findall(r'"sha1"\s*:\s*"([a-f0-9]{40})"', page.replace('\\"', '"')))
    if len(values) != 1:
        raise ValueError('No unique published file SHA-1 in Box preview')
    return values.pop()


def discover_sources():
    with urllib.request.urlopen(FOLDER, timeout=90) as response:
        entries = parse_box_listing(response.read().decode())
    for entry in entries:
        entry['public_file_page'] = f'https://iastate.box.com/s/{SHARE}/file/{entry["file_id"]}'
        with urllib.request.urlopen(entry['public_file_page'], timeout=90) as response:
            entry['expected_sha1'] = box_sha1(response.read().decode())
    return dict(source_folder=FOLDER, files=entries)


def file_hashes(path):
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            sha1.update(block); sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest()


def download_entry(entry, directory, *, progress=print):
    """Never replace a complete input; verify existing and resumed content."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/entry['name']
    if target.exists():
        sha1, sha256 = file_hashes(target)
        if target.stat().st_size != entry['bytes'] or sha1 != entry['expected_sha1']:
            raise ValueError(f'Existing input differs from published source: {target}')
        return dict(existing=True, seconds=0., sha1=sha1, sha256=sha256)
    partial = target.with_name(target.name+'.part')
    parameters = dict(rm='box_download_shared_file', shared_name=SHARE,
                      file_id='f_'+entry['file_id'])
    url = 'https://iastate.box.com/index.php?'+urllib.parse.urlencode(parameters)
    started = time.monotonic()
    for attempt in range(3):
        try:
            current = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'maize-regulatory-census/0.1'}
            if current:
                headers['Range'] = f'bytes={current}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = response.status == 206 and current > 0
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {current}-'):
                    raise ValueError('Resume range mismatch')
                if not append:
                    current = 0
                last_report = time.monotonic()
                with partial.open('ab' if append else 'wb') as handle:
                    while block := response.read(4*1024*1024):
                        handle.write(block); current += len(block)
                        if current > entry['bytes']:
                            raise ValueError('Download exceeds published byte count')
                        if progress and time.monotonic()-last_report >= 20:
                            progress(f'{entry["name"]}: {current/1e6:.1f}/{entry["bytes"]/1e6:.1f} MB', flush=True)
                            last_report = time.monotonic()
            if current != entry['bytes']:
                raise OSError('Incomplete download; retain partial for resumption')
            sha1, sha256 = file_hashes(partial)
            if sha1 != entry['expected_sha1']:
                raise ValueError('Downloaded content does not match published SHA-1')
            partial.replace(target)
            receipt = dict(**entry, source_folder=FOLDER, download_url='https://iastate.box.com/index.php',
                           download_parameters=parameters, sha1=sha1, sha256=sha256,
                           published_checksum_matches=True, seconds=time.monotonic()-started)
            target.with_name(target.name+'.download.json').write_text(json.dumps(receipt, indent=2)+'\n')
            if progress:
                progress(f'Verified {entry["name"]} in {receipt["seconds"]:.1f}s', flush=True)
            return receipt
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            if progress:
                progress(f'Transfer interrupted; resuming {entry["name"]}', flush=True)
    raise AssertionError('Unreachable')
