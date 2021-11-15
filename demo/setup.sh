mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor='#ffffff'\n\
backgroundColor='#bc3e22'\n\
secondaryBackgroundColor='#6e757c'\n\
textColor='#ffffff'\n\
\n\
[deprecation]\n\
showPyplotGlobalUse=false\n\
\n\
" > ~/.streamlit/config.toml