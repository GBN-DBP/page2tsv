import json
import glob
import re
import os
from io import StringIO
from pathlib import Path

import numpy as np
import click
import pandas as pd
import requests
from lxml import etree as ET
from urllib.parse import quote

from ocrd_models.ocrd_mets import OcrdMets
from ocrd_models.ocrd_page import parse
from ocrd_modelfactory import page_from_file
from ocrd_utils import bbox_from_points

from .ned import ned
from .ner import ner
from .tsv import read_tsv, write_tsv, extract_doc_links
from .ocr import get_conf_color


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('url-file', type=click.Path(exists=False), required=True, nargs=1)
def extract_document_links(tsv_file, url_file):

    parts = extract_doc_links(tsv_file)

    urls = [part['url'] for part in parts]

    urls = pd.DataFrame(urls, columns=['url'])

    urls.to_csv(url_file, sep="\t", quoting=3, index=False)


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('annotated-tsv-file', type=click.Path(exists=False), required=True, nargs=1)
def annotate_tsv(tsv_file, annotated_tsv_file):

    parts = extract_doc_links(tsv_file)

    annotated_parts = []

    for part in parts:

        part_data = StringIO(part['header'] + part['text'])

        df = pd.read_csv(part_data, sep="\t", comment='#', quoting=3)

        df['url_id'] = len(annotated_parts)

        annotated_parts.append(df)

    df = pd.concat(annotated_parts)

    df.to_csv(annotated_tsv_file, sep="\t", quoting=3, index=False)


@click.command()
# @click.argument('page-xml-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('mets-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tsv-out-file', type=click.Path(), required=True, nargs=1)
@click.option('--file-grp', type=str, required=True)
@click.option('--purpose', type=click.Choice(['fonts', 'skew'], case_sensitive=False), default="fonts",
              help="Purpose of output tsv file. "
                   "\n\nNERD: NER/NED application/ground-truth creation. "
                   "\n\nOCR: OCR application/ground-truth creation. "
                   "\n\ndefault: NERD.")
@click.option('--image-url', type=str, default='http://empty')
@click.option('--ner-rest-endpoint', type=str, default=None,
              help="REST endpoint of sbb_ner service. See https://github.com/qurator-spk/sbb_ner for details. "
                   "Only applicable in case of NERD.")
@click.option('--ned-rest-endpoint', type=str, default=None,
              help="REST endpoint of sbb_ned service. See https://github.com/qurator-spk/sbb_ned for details. "
                   "Only applicable in case of NERD.")
@click.option('--noproxy', type=bool, is_flag=True, help='disable proxy. default: enabled.')
@click.option('--scale-factor', type=float, default=1.0, help='default: 1.0')
@click.option('--ned-threshold', type=float, default=None)
@click.option('--min-confidence', type=float, default=None)
@click.option('--max-confidence', type=float, default=None)
@click.option('--ned-priority', type=int, default=1)
@click.option('--scheme', type=str, default='http')
@click.option('--server', type=str, required=True)
@click.option('--prefix', type=str, default='')
@click.option('--segment-type', type=str, default='Page')
# def page2tsv(page_xml_file, tsv_out_file, purpose, image_url, ner_rest_endpoint, ned_rest_endpoint,
def page2tsv(mets_file, tsv_out_file, file_grp, purpose, image_url, ner_rest_endpoint, ned_rest_endpoint,
             noproxy, scale_factor, ned_threshold, min_confidence, max_confidence, ned_priority, scheme, server,
             prefix, segment_type):
    if purpose == "fonts":
        # out_columns = [
        #     'text_equiv', 'conf', 'language', 'font_family', 'font_size', 'bold', 'italic', 'letter_spaced',
        #     'segment_type', 'segment_id', 'url_id', 'region', 'rotation'
        # ]
        out_columns = [
            'text_equiv', 'language', 'font_family', 'segment_type', 'segment_id', 'url_id', 'region', 'rotation',
            'full_region', 'full_rotation'
        ]
    elif purpose == "skew":
        out_columns = ['segment_type', 'segment_id', 'url_id', 'region', 'rotation']
    else:
        raise RuntimeError("Unknown purpose.")

    if noproxy:
        os.environ['no_proxy'] = '*'

    pd.DataFrame([], columns=out_columns).to_csv(tsv_out_file, sep="\t", quoting=3, index=False)

    tsv = []
    urls = []

    # base_url = "scheme://server/prefix/identifier/region/size/rotation/quality.format"
    base_url = "scheme://server/prefix/identifier"

    base_url = re.sub('scheme', scheme, base_url)
    base_url = re.sub('server', server, base_url)

    if prefix:
        base_url = re.sub('prefix', prefix, base_url)
    else:
        base_url = re.sub('/prefix', '', base_url)

    mets = OcrdMets(filename=mets_file)

    basedir = Path(mets_file).parents[0]

    for fl in mets.find_files(fileGrp=file_grp):
        fl.local_filename = str(basedir / Path(fl.local_filename))

        pcgts = page_from_file(fl)
        page = pcgts.get_Page()

        # pcgts = parse(page_xml_file)
        # page = pcgts.get_Page()

        urls.append(re.sub('identifier', quote(page.get_imageFilename(), safe=''), base_url))

        region = "full"
        try:
            rotation = page.get_orientation()
        except:
            rotation = 0.0

        page_id = fl.pageId
        url_id = len(urls) - 1

        if segment_type == 'Page':
            rotation = str(rotation % 360)

            if purpose == 'skew':
                tsv.append((segment_type, page_id, url_id, region, rotation))
        else:
            segments = []

            if 'Region' in segment_type:
                for reg in page.get_AllRegions(classes=[re.sub('Region', '', segment_type)], order='reading-order'):
                    try:
                        region_rotation = rotation + reg.get_orientation()
                    except:
                        region_rotation = rotation
                    segments.append((reg, region_rotation, page, rotation))
            else:
                regions = page.get_AllRegions(classes=['Text'], order='reading-order')
                lines = []
                for reg in regions:
                    try:
                        region_rotation = rotation + reg.get_orientation()
                    except:
                        region_rotation = rotation
                    for line in reg.get_TextLine():
                        try:
                            line_rotation = region_rotation + line.get_orientation()
                        except:
                            line_rotation = region_rotation
                        lines.append((line, line_rotation, reg, region_rotation))
                if segment_type == 'TextLine':
                    segments = lines
                else:
                    words = []
                    for line, line_rotation, reg, region_rotation in lines:
                        for word in line.get_Word():
                            try:
                                word_rotation = line_rotation + word.get_orientation()
                            except:
                                word_rotation = line_rotation
                            words.append((word, word_rotation, reg, region_rotation))
                    segments = words

            for segment, rotation, full, full_rotation in segments:
                coords = segment.get_Coords()
                x0, y0, x1, y1 = bbox_from_points(coords.points)

                region = str(x0) + ',' + str(y0) + ',' + str(x1 - x0) + ',' + str(y1 - y0)
                rotation = str(rotation % 360)

                full_coords = full.get_Coords()
                x0, y0, x1, y1 = bbox_from_points(full_coords.points)

                full_region = str(x0) + ',' + str(y0) + ',' + str(x1 - x0) + ',' + str(y1 - y0)
                full_rotation = str(full_rotation % 360)

                segment_id = page_id + '_' + segment.get_id()

                if purpose == 'fonts':
                    try:
                        text_equivs = [
                            (text_equiv.get_Unicode(), text_equiv.get_conf()) for text_equiv in segment.get_TextEquiv()
                        ]

                        try:
                            language = segment.get_language()
                            if not language:
                                language = '-'
                        except:
                            language = '-'

                        try:
                            text_style = segment.get_TextStyle()

                            try:
                                font_family = text_style.get_fontFamily()
                                if not font_family:
                                    font_family = '-'
                            except:
                                font_family = '-'

                            try:
                                font_size = text_style.get_fontSize()
                                if not font_size:
                                    font_size = '-'
                            except:
                                font_size = '-'

                            try:
                                bold = text_style.get_bold()
                                if not bold:
                                    bold = '-'
                            except:
                                bold = '-'

                            try:
                                italic = text_style.get_italic()
                                if not italic:
                                    italic = '-'
                            except:
                                italic = '-'

                            try:
                                letter_spaced = text_style.get_letterSpaced()
                                if not letter_spaced:
                                    letter_spaced = '-'
                            except:
                                letter_spaced = '-'
                        except:
                            font_family = '-'
                            font_size = '-'
                            bold = '-'
                            italic = '-'
                            letter_spaced = '-'

                        for text_equiv, conf in text_equivs:
                            tsv.append(
                                (
                                    text_equiv,
                                    # conf,
                                    language,
                                    font_family,
                                    # font_size,
                                    # bold,
                                    # italic,
                                    # letter_spaced,
                                    segment_type,
                                    segment_id,
                                    url_id,
                                    region,
                                    rotation,
                                    full_region,
                                    full_rotation
                                )
                            )
                    except:
                        pass
                else:
                    tsv.append((segment_type, segment_id, url_id, region, rotation))

    tsv = pd.DataFrame(tsv, columns=out_columns)

    if len(tsv) == 0:
        return

    with open(tsv_out_file, 'a') as f:
        # f.write('# ' + image_url + '\n')
        for url in urls:
            f.write('# ' + url + '\n')

    tsv = tsv[out_columns].reset_index(drop=True)

    try:
        tsv.to_csv(tsv_out_file, sep="\t", quoting=3, index=False, mode='a', header=False)
    except requests.HTTPError as e:
        print(e)


@click.command()
@click.option('--output-filename', '-o', help="Output filename. "
                                              "If omitted, PAGE-XML filename with .corrected.xml extension")
@click.option('--keep-words', '-k', is_flag=True, help="Keep (out-of-date) Words of TextLines")
@click.argument('page-file')
@click.argument('tsv-file')
def tsv2page(output_filename, keep_words, page_file, tsv_file):
    if not output_filename:
        output_filename = Path(page_file).stem + '.corrected.xml'
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)
    tree = ET.parse(page_file)
    for _, row in tsv.iterrows():
        el_textline = tree.find(f'//pc:TextLine[@id="{row.line_id}"]', namespaces=ns)
        el_textline.find('pc:TextEquiv/pc:Unicode', namespaces=ns).text = row.TEXT
        if not keep_words:
            for el_word in el_textline.findall('pc:Word', namespaces=ns):
                el_textline.remove(el_word)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(ET.tostring(tree, pretty_print=True).decode('utf-8'))


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tsv-out-file', type=click.Path(), required=True, nargs=1)
@click.option('--ner-rest-endpoint', type=str, default=None,
              help="REST endpoint of sbb_ner service. See https://github.com/qurator-spk/sbb_ner for details.")
@click.option('--ned-rest-endpoint', type=str, default=None,
              help="REST endpoint of sbb_ned service. See https://github.com/qurator-spk/sbb_ned for details.")
@click.option('--ned-json-file', type=str, default=None)
@click.option('--noproxy', type=bool, is_flag=True, help='disable proxy. default: proxy is enabled.')
@click.option('--ned-threshold', type=float, default=None)
@click.option('--ned-priority', type=int, default=1)
def find_entities(tsv_file, tsv_out_file, ner_rest_endpoint, ned_rest_endpoint, ned_json_file, noproxy, ned_threshold,
                  ned_priority):

    if noproxy:
        os.environ['no_proxy'] = '*'

    tsv, urls = read_tsv(tsv_file)

    try:
        if ner_rest_endpoint is not None:

            tsv, ner_result = ner(tsv, ner_rest_endpoint)

        elif os.path.exists(tsv_file):

            print('Using NER information that is already contained in file: {}'.format(tsv_file))

            tmp = tsv.copy()
            tmp['sen'] = (tmp['No.'] == 0).cumsum()
            tmp.loc[~tmp['NE-TAG'].isin(['O', 'B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'I-LOC', 'I-ORG']), 'NE-TAG'] = 'O'

            ner_result = [[{'word': str(row.TOKEN), 'prediction': row['NE-TAG']} for _, row in sen.iterrows()]
                          for _, sen in tmp.groupby('sen')]
        else:
            raise RuntimeError("Either NER rest endpoint or NER-TAG information within tsv_file required.")

        if ned_rest_endpoint is not None:

            tsv, ned_result = ned(tsv, ner_result, ned_rest_endpoint, json_file=ned_json_file, threshold=ned_threshold,
                                  priority=ned_priority)

            if ned_json_file is not None and not os.path.exists(ned_json_file):

                with open(ned_json_file, "w") as fp_json:
                    json.dump(ned_result, fp_json, indent=2, separators=(',', ': '))

        write_tsv(tsv, urls, tsv_out_file)

    except requests.HTTPError as e:
        print(e)


@click.command()
@click.option('--xls-file', type=click.Path(exists=True), default=None,
              help="Read parameters from xls-file. Expected columns:  Filename, iiif_url, scale_factor.")
@click.option('--directory', type=click.Path(exists=True), default=None,
              help="Search directory for PPN**/*.xml files. Extract PPN and file number into image-url.")
@click.option('--purpose', type=click.Choice(['NERD', 'OCR'], case_sensitive=False), default="NERD",
              help="Purpose of output tsv file. "
                   "\n\nNERD: NER/NED application/ground-truth creation. "
                   "\n\nOCR: OCR application/ground-truth creation. "
                   "\n\ndefault: NERD.")
def make_page2tsv_commands(xls_file, directory, purpose):
    if xls_file is not None:

        if xls_file.endswith(".xls"):
            df = pd.read_excel(xls_file)
        else:
            df = pd.read_excel(xls_file, engine='openpyxl')

        df = df.dropna(how='all')

        for _, row in df.iterrows():
            print('page2tsv $(OPTIONS) {}.xml {}.tsv --image-url={} --scale-factor={} --purpose={}'.
                  format(row.Filename, row.Filename, row.iiif_url.replace('/full/full', '/left,top,width,height/full'),
                         row.scale_factor, purpose))

    elif directory is not None:
        for file in glob.glob('{}/**/*.xml'.format(directory), recursive=True):

            ma = re.match('(.*/(PPN[0-9X]+)/.*?([0-9]+).*?).xml', file)

            if ma:
                print('page2tsv {} {}.tsv '
                      '--image-url=https://content.staatsbibliothek-berlin.de/dc/'
                      '{}-{:08d}/left,top,width,height/full/0/default.jpg --scale-factor=1.0 --purpose={}'.
                      format(file, ma.group(1), ma.group(2), int(ma.group(3)), purpose))

