"""Render the shared reference content as an offline, searchable PDF."""
from pathlib import Path
from html import escape
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

HERE=Path(__file__).resolve().parent
font=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('Reference','DejaVuSans.ttf'),('ReferenceBold','DejaVuSans-Bold.ttf'),('ReferenceMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(font/file)))
styles={
 'title':ParagraphStyle('title',fontName='ReferenceBold',fontSize=23,leading=29,spaceAfter=12,keepWithNext=True),
 'subtitle':ParagraphStyle('subtitle',fontName='Reference',fontSize=10,leading=14,textColor=colors.HexColor('#647282'),spaceAfter=18),
 'h1':ParagraphStyle('section',fontName='ReferenceBold',fontSize=18,leading=23,spaceBefore=22,spaceAfter=10,keepWithNext=True,textColor=colors.HexColor('#087e80')),
 'h2':ParagraphStyle('subsection',fontName='ReferenceBold',fontSize=13,leading=17,spaceBefore=15,spaceAfter=7,keepWithNext=True),
 'p':ParagraphStyle('body',fontName='Reference',fontSize=10.5,leading=15,spaceAfter=9),
 'label':ParagraphStyle('label',fontName='ReferenceBold',fontSize=10.5,leading=15,spaceAfter=7,keepWithNext=True),
 'small':ParagraphStyle('small',fontName='Reference',fontSize=8.5,leading=12,spaceAfter=12,textColor=colors.HexColor('#647282')),
 'cell':ParagraphStyle('cell',fontName='Reference',fontSize=8.5,leading=12),
 'head':ParagraphStyle('head',fontName='ReferenceBold',fontSize=8.5,leading=12,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='ReferenceMono',fontSize=8.5,leading=12,spaceBefore=5,spaceAfter=10),
}
blocks=json.loads((HERE/'presenter_reference.json').read_text())
story=[]
for b in blocks:
    kind=b['kind'];text=escape(b['text'])
    if kind=='table':
        story.append(Paragraph(escape(b['title']),styles['h2']))
        data=[[Paragraph(escape(v),styles['head']) for v in b['headers']]]
        data += [[Paragraph(escape(v),styles['cell']) for v in row] for row in b['rows']]
        table=Table(data,colWidths=[135,105,126,126],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172c45')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f5')]),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#c7d3d4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([table,Spacer(1,12)])
    elif kind=='link':story.append(Paragraph(f'<link href="{escape(b["url"],quote=True)}" color="#087e80">{text}</link>',styles['p']))
    elif kind=='code':
        # Wrap long commands while preserving a fully copyable original in HTML.
        import textwrap
        lines=[]
        for line in b['text'].splitlines():lines += textwrap.wrap(line,92,break_long_words=False,break_on_hyphens=False) or ['']
        story.append(Preformatted('\n'.join(lines),styles['code']))
    else:story.append(Paragraph(text,styles[kind]))

def footer(canvas,doc):
    canvas.setFont('Reference',8);canvas.setFillColor(colors.HexColor('#647282'))
    canvas.drawString(54,28,'REWT · Presenter reference · October 28, 2026')
    canvas.drawRightString(558,28,str(doc.page))

doc=SimpleDocTemplate(str(HERE/'presenter_reference.pdf'),pagesize=(612,792),leftMargin=60,rightMargin=60,topMargin=46,bottomMargin=48,title='REWT — Presenter reference',author='Kenju Tomita')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(HERE/'presenter_reference.pdf')
