"""Render the course proposal to PDF; requires reportlab and DejaVu fonts."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle, Spacer

ROOT=Path(__file__).resolve().parent
FONTS=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('Body','DejaVuSerif.ttf'),('BodyBold','DejaVuSerif-Bold.ttf'),
                  ('BodyItalic','/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'),('Math','DejaVuSans.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
body=ParagraphStyle('Body',fontName='Body',fontSize=10.6,leading=13.1,spaceAfter=5.5)
heading=ParagraphStyle('Heading',parent=body,fontName='BodyBold',fontSize=12.3,leading=14.5,spaceBefore=8,spaceAfter=6,keepWithNext=True)
title=ParagraphStyle('Title',parent=body,fontName='BodyBold',fontSize=16,leading=19,alignment=TA_CENTER,spaceAfter=9)
author=ParagraphStyle('Author',parent=body,fontSize=10.3,leading=13.2,alignment=TA_CENTER,spaceAfter=10)
mathstyle=ParagraphStyle('Math',parent=body,fontName='Math',fontSize=10.2,leading=14,alignment=TA_CENTER,spaceBefore=2,spaceAfter=7)
cell=ParagraphStyle('Cell',parent=body,fontSize=9.8,leading=11.8,spaceAfter=0)

def inline(s):
    substitutions={
        r'\(x_{\leq t}\)':'x<sub>≤t</sub>',
        r'\(z^{(r)}\)':'z<super>(r)</super>',
        r'\(p^{(r)}\)':'p<super>(r)</super>',
    }
    s=escape(s)
    for old,new in substitutions.items():s=s.replace(escape(old),new)
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\*([^*]+)\*',r'<i>\1</i>',s)
    def link(m):
        if m[2].startswith('../'):return m[1]+' (included in project repository)'
        return '<link href="'+m[2]+'" color="#204c78">'+m[1]+'</link>'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)

s=(ROOT/'PROPOSAL.md').read_text();bodytext=s[s.index('## Abstract'):]
parts=bodytext.splitlines();story=[Paragraph('Full Energy-Based Recurrent Transformers<br/>for Sequence Prediction',title),
    Paragraph('Kenju Tomita · kt7432@g.rit.edu<br/>Rochester Institute of Technology<br/>CSCI 635: Introduction to Machine Learning<br/>Individual Project Proposal · October 9, 2026',author)]
paragraph=[];table=[];math=[];in_math=False

def flush():
    if paragraph:
        story.append(Paragraph(inline(' '.join(paragraph)),body));paragraph.clear()
def flush_table():
    if not table:return
    rows=[[Paragraph(inline(c),cell) for c in row] for row in table]
    t=Table(rows,colWidths=[112,378],hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.6,colors.black),
        ('LINEBELOW',(0,0),(-1,0),.4,colors.grey),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7)]))
    story.append(t);story.append(Spacer(1,7));table.clear()
for line in parts:
    if line==r'\[':flush();in_math=True;math=[];continue
    if line==r'\]':
        if math[0].startswith('p^'):
            equation=('p<super>(r)</super> = softmax(z<super>(r)</super>), '
                      'E<sub>θ</sub>(x<sub>≤t</sub>, p<super>(r)</super>) ∈ ℝ')
        elif math[0].startswith('z^'):
            equation=('z<super>(r+1)</super> = z<super>(r)</super> − η ∇<sub>z<super>(r)</super></sub> '
                      'E<sub>θ</sub>(x<sub>≤t</sub>, p<super>(r)</super>)')
        else:
            raise ValueError('Unrecognized proposal equation; provide an explicit rendering')
        story.append(Paragraph(equation,mathstyle));in_math=False;continue
    if in_math:math.append(line);continue
    if line.startswith('## '):
        flush();flush_table();label=line[3:]
        if label.startswith(('2.','3.')) or label=='References':story.append(PageBreak())
        story.append(Paragraph(inline(label),heading));continue
    if line.startswith('|'):
        flush();cells=[c.strip() for c in line.strip('|').split('|')]
        if not all(set(c)<=set('-: ') for c in cells):table.append(cells)
        continue
    flush_table()
    if re.match(r'^\d+\. ',line):
        flush();story.append(Paragraph(inline(line),body))
    elif not line:flush()
    else:paragraph.append(line)
flush();flush_table()
def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Body',9)
    canvas.drawString(61,30,'Kenju Tomita | CSCI 635 | Project Proposal')
    canvas.drawRightString(551,30,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(ROOT/'proposal.pdf'),pagesize=letter,leftMargin=61,rightMargin=61,
    topMargin=43,bottomMargin=45,title='Full Energy-Based Recurrent Transformers for Sequence Prediction',author='Kenju Tomita')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
