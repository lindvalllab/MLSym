import spacy
import re
import argparse

# Recommended start and end intervals for HPI selection from a clinical note
start_interval = ['[ ?y.]{0,6}(hpi\W{0,3}|)interval (history|hx)',
                  '[ ?y.]{0,6}interim history',
                  '[ ?y.]{0,6}change(s|) since last visit',
                  '[ ?y.]{0,6}at today\'s visit',
                  '^[ ?y.]{0,6}current status']
start_hpi = ['hpi', 'history of present illness']
start_cc = ['^[ ?y.]{0,2}diagnosis/identification',
            'identif(ication|ier|ying information)',
            '(cc/|patient |)id(/cc| and cc):',
            'reason(s|) for (the |)(visit|consult(ation|))',
            'chief complaint','c(/|)c:',
            'today']
stop_sec = [#'^[ ?y.]{0,6}(history of present illness|hpi)',
            '^[ ?y.]{0,6}oncolog(y|ic(al|)) (and other medical |)history', 'oncolog(y|ic(al|)) (and other medical |)history:',
            '^[ ?y.]{0,6}\(copied from .{1,15} for clinical reference', '^[ ?y.]{0,6}medical oncology visit',
            '^[ ?y.]{0,6}(onc|gyn(ecologic|)|bone health fall|menstrual( and obstetric|)|ob(stetric|)|psychiatric) history',
            '^[ ?y.]{0,6}(past|prior)( medical|) .{0,20}history', '^[ ?y.]{0,6}pm(d|h)', 'pmh(/psh|):',
            '^[ ?y.]{0,6}(personal and |)(social|family|desensitization) (history|hx:)',
            '^[ ?y.]{0,6}problem(s|)( list|).{1,3}',
            '^[ ?y.]{0,6}(patient|) active problem list',
            '^[ ?y.]{0,6}relevant .{0,15}history', '^[ ?y.]{0,6}relevant past .{0,20}history',
            #'^[ ?y.]{0,6}review of (systems|symptoms)', '^[ ?y.]{0,6}ros was negative for', '[ ?y.]{0,6}(ros|constitutional):',
            '^[ ?y.]{0,6}physical exam(ination|)',
            '^[ ?y.]{0,6}lab(s:| result)',  '^[ ?y.]{0,6}latest known visit with result', '^[ ?y.]{0,6}blood Draw',
            'component\s+date\s+value\s+ref', 'component\s+value\s+date', 'result\s+value\s+ref\s+range',
            '^[ ?y.]{0,6}imaging( studies| result(s|)|):', '^[ ?y.]{0,6}.{0,15}\sstudies:', '^[ ?y.]{0,6}pathology(/biopsy|):',
            '^[ ?y.]{0,6}(pathologic |)diagnosis', '^[ ?y.]{0,6}diagnosis:',
            '^[ ?y.]{0,6}allerg(ies:|en reactions)', '^[ ?y.]{0,6}medication((s|):| sig)', '^[ ?y.]{0,6}medications reviewed', '^[ ?y.]{0,6}drug/med allergies',
            '^[ ?y.]{0,6}current (med(ication|)s:|outpatient (prescription|medication))', '^[ ?y.]{0,6}current .{0,10}(treatments/ |)medications',
            '^[ ?y.]{0,6}controlled substance information', '^[ ?y.]{0,6}immunizations:',
            '^[ ?y.]{0,6}interventions(:| tried for .{1,15}:)',
            '^[ ?y.]{0,6}problems, past medical history',
            '^[ ?y.]{0,6}information obtained from\s','^[ ?y.]{0,6}information from collateral',
            '^[ ?y.]{0,6}stage [1-4]$', '[ ?y.]{0,6}active therapy at visit .{0,15}',
            '^[ ?y.]{0,6}symptom assessment:', 'assessment (&|and) plan',
            '^[ ?y.]{0,6}review of patient\'s current problem list',
            '^[ ?y.]{0,6}oncopanel', '[ ?y.]{0,6}treatment (summary|plan):','[ ?y.]{0,6}active treatment (& therapy |)days',
            '^[ ?y.]{0,6}requesting physician', 'staging form:', '--hospital course--', '\W.{1,15} symptom scale:',
            '^[ ?y.]{0,6}answers for hpi/ros submitted by the patient',
            '^[ ?y.]{0,6}i have reviewed and confirmed', '24 hour event']

nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser', 'tagger','lemmatizer'])
nlp.create_pipe('sentencizer')
nlp.add_pipe('sentencizer')

def extract_hpi(text):
    """
    Extract HPI from clinical text via start and end defined above

    :return: HPI text
    :rtype: text

    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    inter = any(re.search(s.replace('[','[\n\r'), doc.text.lower()) for s in start_interval)
    out=[]
    start=2
    hpi,cc=False,False
    for idx, sent in enumerate(sentences):
        include=False
        if hpi!='done':
            if cc and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in stop_sec + start_interval + start_hpi): cc = False
            if hpi or cc:
                if any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in stop_sec): hpi = 'done'
                include = True
            else:
                if any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_cc): include = True; cc = True
                if (inter and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_interval)) or \
                                        (not inter and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_hpi)): include = True; hpi = True
    #     length = len(sent.split())
        if include:
    #         start += length
            out.append(sent)
    #     idx_map.insert(start, -1)
    #         start += 1
    return ' '.join(i for i in out)


def get_hpi_output(hpi_input_data,col_name):
    """
    Place the HPI outputs into a dataframe and select required provider department

    :return: dictionary of note id and HPI
    :rtype: dict

    """
    hpi_input_data['HPI'] = hpi_input_data[col_name].apply(lambda x: extract_hpi(x))
    output = hpi_input_data[['HPI']].rename(columns={'HPI': 'text'}).to_dict('records')
    # with open(op_filepath, 'w') as fp:
    #     json.dump(output, fp)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract hpi from clinical text')
    parser.add_argument('-ipf', '--input_file', dest='ip', help='Path to input file', required=True)
    parser.add_argument('-cn', '--column_name', dest='cn', help='Name of the column containing the note', required=True)
    args = parser.parse_args()

    hpi_output = get_hpi_output(args.ip)
