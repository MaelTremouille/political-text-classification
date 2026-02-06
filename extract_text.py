from arkindex_export import open_database, Element, Metadata, Transcription
from arkindex_export.queries import list_children
from pathlib import Path
from tqdm import tqdm

# load the  export
open_database(Path("sciencespo-archelec-20260129-223203.sqlite"))
# create a folder to store the text files
TEXT_FOLDER = "text_files"
output_folder = Path(TEXT_FOLDER)
output_folder.mkdir(exist_ok=True)
for year in ['1981']:
    for type in ['legislatives', 'presidentielle']:
        # create the folder
        year_folder = output_folder / year
        year_folder.mkdir(exist_ok=True)
        type_folder = year_folder / type
        type_folder.mkdir(exist_ok=True)

# compute some statistics
print("Number of folders", Element.select().where(Element.type == 'folder').count())
print("Number of pages:", Element.select().where(Element.type == 'page').count())

legislatives_1981_id = 'd51ea3db-68ee-4cc0-a87f-736ee17c5f87'
#presidentielle_1981_id = '4192aaa9-8485-433a-b0e3-559d2259e067'
year = '1981'
type = 'legislatives'
# list all documents in legislative_1981_id
documents = list_children(legislatives_1981_id).where(Element.type == 'document')

# number of documents
print("Number of documents", documents.count())
transcriptions_number = 0
# for each document, list the direct children page, for each page extract the transcription , concatenate the transcription and save in a text file named as the document
for document in tqdm(documents):
    pages = list_children(document.id).where(Element.type == 'page')
    transcriptions = ""
    for page in pages:
        print(page.id)
        page_transcription = Transcription.select().where(Transcription.element == page.id).first()
        if page_transcription:
            transcriptions += page_transcription.text

    if transcriptions:
        with open(f"{TEXT_FOLDER}/{year}/{type}/{document.name}.txt", "w") as f:
            f.write(transcriptions)
        transcriptions_number += 1
print("Number of transcriptions", transcriptions_number)


