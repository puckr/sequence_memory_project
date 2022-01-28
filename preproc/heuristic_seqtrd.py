import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    
    allowed template fields - follow python string module: 
    
    item: index within category 
    subject: participant id 
    seqitem: run number during scanning
    subindex: sub index within group
    """
    
    dwi = create_key('dmri/dwi_{item:03d}', outtype=('dicom', 'nii.gz'))
    t1 = create_key('anatomy/T1_{item:03d}', outtype=('dicom', 'nii.gz'))
    trd_bold = create_key('trd_bold/bold_{item:03d}/trd_bold', outtype=('dicom', 'nii.gz'))
    seq_bold = create_key('seq_bold/bold_{item:03d}/seq_bold', outtype=('dicom', 'nii.gz'))
    info = {dwi: [], t1: [], trd_bold: [], seq_bold: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if (sl == 186) and (nt == 1) and ('T1' in s[12]):
            info[t1].append(s[2])
        elif (nt != 355) and ('TRD' in s[12]):
            info[trd_bold].append(s[2])
            last_run = s[2]
        elif (nt == 355) and ('SEQ' in s[12]):
            info[seq_bold].append(s[2])
            last_run = s[2]
        elif (sl > 1) and ('DTI' in s[12]):
            info[dwi].append(s[2])
        else:
            pass
    return info
