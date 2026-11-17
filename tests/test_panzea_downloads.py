import hashlib
import io
import json

import pytest

from maize_regulatory.panzea_downloads import box_sha1, download_entry, parse_box_listing


def test_public_listing_and_escaped_checksum_parse_without_accepting_missing_chromosomes():
    files = [dict(name=f'panzea_hapmap_282_chr{c}.vcf.gz{s}', id=c*10+len(s), itemSize=100)
             for c in range(1, 11) for s in ['', '.tbi']]
    page = 'Box.postStreamData = '+json.dumps({'nested': {'files': files}})+';ignored'
    entries = parse_box_listing(page)
    assert len(entries) == 20 and entries[0]['chromosome'] == 1
    expected = 'a'*40
    assert box_sha1(json.dumps(json.dumps({'sha1': expected, 'file_version': {'sha1': expected}}))) == expected
    with pytest.raises(ValueError, match='all ten'):
        parse_box_listing('Box.postStreamData = '+json.dumps(files[:-1]))
    with pytest.raises(ValueError, match='unique'):
        box_sha1(json.dumps({'sha1': 'a'*40})+json.dumps({'sha1': 'b'*40}))


def test_download_resumes_a_partial_and_checks_published_content(monkeypatch, tmp_path):
    data = b'example public source file bytes'
    entry = dict(name='source.gz', chromosome=1, file_id='123', bytes=len(data),
                 expected_sha1=hashlib.sha1(data).hexdigest())
    (tmp_path/'source.gz.part').write_bytes(data[:7])

    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': f'bytes 7-{len(data)-1}/{len(data)}'}

    def open_response(request, timeout):
        assert request.get_header('Range') == 'bytes=7-'
        return Response(data[7:])

    monkeypatch.setattr('urllib.request.urlopen', open_response)
    receipt = download_entry(entry, tmp_path, progress=None)
    assert (tmp_path/'source.gz').read_bytes() == data
    assert receipt['published_checksum_matches']
    assert download_entry(entry, tmp_path, progress=None)['existing']
    (tmp_path/'source.gz').write_bytes(b'bad existing content')
    with pytest.raises(ValueError, match='Existing input'):
        download_entry(entry, tmp_path, progress=None)
