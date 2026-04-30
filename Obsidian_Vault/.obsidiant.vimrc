" --- CONFIGURATION DE BASE ---

" Utiliser le presse-papier système (permet de yank/paste entre Obsidian et les autres apps)
set clipboard=unnamed

" Sauter les lignes visuelles (si une ligne est longue et revient à la ligne, j/k descend d'une ligne visuelle)
nmap j gj
nmap k gk

" Effacer le surlignage de recherche avec Esc
nmap <Esc> :nohl


" --- NAVIGATION ---

" Aller au début et à la fin de la ligne plus facilement
nmap H ^
nmap L $

" Raccourcis pour reculer/avancer dans l'historique des notes (comme les flèches du navigateur)
exmap back obcommand app:go-back
nmap <C-o> :back
exmap forward obcommand app:go-forward
nmap <C-i> :forward


" --- ÉDITION & SURROUND (Nécessite le plugin Vimrc Support) ---

" Ajouter des [[ ]] autour d'une sélection (très utile pour les liens Obsidian)
exmap surround_wiki surround [[ ]]
vmap [[ :surround_wiki

" Ajouter des étoiles pour le gras ou l'italique
exmap surround_bold surround ** **
vmap s* :surround_bold

" Supprimer les espaces vides en fin de ligne
nmap _$ :%s/\s\+$//ge<CR>


" --- COMMANDES SPÉCIFIQUES À OBSIDIAN ---

" Mapper la barre d'espace pour ouvrir la palette de commandes
exmap commandPalette obcommand command-palette:open
nmap <Space>p :commandPalette<CR>

" Mapper pour ouvrir/fermer les barres latérales
exmap toggleLeftSidebar app:toggle-left-sidebar
nmap <C-b> :toggleLeftSidebar<CR>

" Sauvegarder (même si Obsidian sauvegarde tout seul, par réflexe)
exmap save obcommand editor:save-file
nmap :w :save