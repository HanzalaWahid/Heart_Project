import sys
from streamlit.web import cli as stcli
import os

def main():
    # Resolve the path to app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    
    # Set arguments for streamlit run
    sys.argv = ["streamlit", "run", app_path]
    
    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
