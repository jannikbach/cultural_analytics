import dotenv
import os

dotenv.load_dotenv()

def get_base_path():
    return os.getenv('CA_BASE_PATH', '../')

HARD_STYLES=['House', 'Hard House', 'Techno', 'Hard Techno', 'Trance', 'Hard Trance']
HARD_STYLES_WITHOUT_HARD=['House', 'Techno', 'Trance']
TOP_STYLES=['House', 'Experimental', 'Synth-pop', 'Techno', 'Ambient', 'Electro', 'Trance', 'Downtempo', 'Disco', 'Tech House']
ALL_STYLES=['House', 'Experimental', 'Synth-pop', 'Techno', 'Ambient', 'Electro', 'Trance', 'Downtempo', 'Disco', 'Tech House', 'Noise', 'Deep House', 'Drum n Bass', 'Progressive House', 'Industrial', 'Euro House', 'Abstract', 'Hardcore', 'Minimal', 'Pop Rock', 'Breakbeat', 'Drone', 'Progressive Trance', 'IDM', 'New Wave', 'Breaks', 'Dark Ambient', 'Dance-pop', 'Hard Trance', 'Electro House']