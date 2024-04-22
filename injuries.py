import requests
import PyPDF2
from datetime import datetime
import re

base_url = f'https://ak-static.cms.nba.com/referee/injury/Injury-Report'



def download_pdf(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open(filename, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Iterate through each page and extract text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    return text

def fetch_injured_players(current_date = None):
    current_date = datetime.now().strftime("%Y-%m-%d")
    dated_url = f'{base_url}_{current_date}'
    last_url = None

    for time in ['12AM', '01AM', '02AM', '03AM', '04AM', '05AM', '06AM', '07AM', '08AM', '09AM', '10AM', '11AM', '12PM', '01PM', '02PM', '03PM', '04PM', '05PM', '06PM', '07PM', '08PM', '09PM', '10PM', '11PM']:
        full_url = f'{dated_url}_{time}.pdf'
        try:
            response = requests.get(full_url)

            response.raise_for_status()
            last_url = full_url
        except:
            try:
                full_url = f'{dated_url}_11PM.pdf'
                download_pdf(last_url, "injury_report.pdf")

                text = extract_text_from_pdf('injury_report.pdf')
                text = text.replace('\n', ' ').replace('Injury Report:', '').replace('G League', 'G-League')

                # Updated regex pattern
                pattern = r"([A-Za-z'.]+(?:\sJr\.|\sSr\.|\sII|\sIII|\sIV)?),\s*([A-Za-z'.]+)\s*(Available|Out|Doubtful|Questionable|Probable|Not With Team)\s*(Injury/Illness|G-League|Two-Way|Coach's Decision|League Suspension)"


                matches = re.findall(pattern, text)

                injured_players = []
                for match in matches:

                    if match[2] == 'Out' or match[2] == 'Doubtful':
                        injured_players.append(f'{match[1]} {match[0]}')
                return injured_players
            except:
                return []




print(fetch_injured_players())
