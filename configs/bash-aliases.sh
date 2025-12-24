# Bash Aliases and Configuration
# Add to: ~/.bashrc

# Aliases útiles
alias ll="ls -la"
alias gs="git status"
alias gp="git push"
alias gc="git commit"
alias gd="git diff"
alias docker-clean="docker system prune -af"

# uv/Python
source $HOME/.local/bin/env

# PATH
export PATH="$HOME/.local/bin:$PATH"

# Jupyter Lab desde proyecto
alias jlab="cd ~/projects/datascience && uv run jupyter lab --ip=0.0.0.0"

# Activar entorno de proyecto
alias ds="cd ~/projects/datascience"