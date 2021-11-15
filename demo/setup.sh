mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor='#ffffff'\n\
backgroundColor='#d43614'\n\
secondaryBackgroundColor='#3b3d3d'\n\
textColor='#ffffff'\n\
\n\
[deprecation]\n\
showPyplotGlobalUse=false\n\
\n\
" > ~/.streamlit/config.toml
apt install graphviz