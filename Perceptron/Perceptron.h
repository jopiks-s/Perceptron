
�}|�r�|d dk�r�|d dk	�r�|d dk	�r�d\}}|d |d  }}t|d| jd���"}|�� }t|�}d}i }	�x ||k �r�|| j }
t |||
� �}g }xf||
|
| � D ]R}y|�	|	| � W n8   t
dt|� | j d � |	|< }|�	|� Y nX �qW |
d | }d �|�}t�||��r�d}|dk�r�|dk�r�t|�dd��dd�� t�||��r�d}d}�q�W W dQ R X dS )r   Nr(   )r)   r   i   r   �ascii�ignore)r   r   �/�   �,�   )FFTF)�intr   r   r*   r+   r
   r,   r1   r   r.   r/   r0   r	   r2   r   �encode�decoder   r   �split)r   r3   Zsearch_patternZtop_nZstart_end_delimiterr4   r5   Zlen_contentsr#   Z
encstr_map�constZnocZ	dec_charsr7   Zdec_cZ
print_lineZmatched_cntrZstart_end_elementsZstart_counterZend_counterZstart_patternZend_patternr   r   r   �unobfuscateH   s�     
"
*

"

 

"
,

0
"
 zCgrep.unobfuscatec             C   s�  t tj�dkr| ��  dS tjd dkrxtjd dkrNtjd tjd  }}ntjd tjd  }}| �|d d|� �n*tjd dkr�| �tjd � �nt tj�dkr�| �tjd tjd � n�t tj�dkr�tjd d	kr�n�t tj�dk�rtjd d
k�rn�t tj�dk�rTtjd d
k�r2tjd d	k�s�tjd d	k�rTtjd d
k�rTnNt tj�dk�r�tjd d	k�r�| �tjd tjd tjd � n| ��  dS dS )r   r   z-sr<   z-nr>   r   r   z-ez-cz-i�   N)r1   r   r   r   rD   r8   )r   ZsedArrayr3   r   r   r   �main�   s,     P "z
Cgrep.main)Nr   N)
�__name_