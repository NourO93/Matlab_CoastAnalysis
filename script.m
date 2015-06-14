%{

%SCOPO-> creazione di uno script che usi kmeans per segmentare un'immagine satellitare in
regioni, per estrazione di pixel appartenenti alla costa. Analisi del grado
di irregolarità di questa e dei punti di maggiore difetto di convessità
(golfi)

USO:
    -settare il path dell'immagine per la variabile path_immagine
    -settare 1 in flag_analisi_conv per abilitare l'analisi di convessità
    dei contorni
    -avviare lo script -> verranno mostrati i colori dei punti costieri a
    seconda del grado di variabilità. Una pressione di un tasto, avvierà la
    rappresentazione dei punti di maggiore difetto di convessità .
%}

clear
close all

%-----------------parametri-------------
flag_analisi_conv = 1;
path_immagine = '7.jpg';
%parametri per calcolo della variabilità gradiente pixel bordi
sizeKern = 6; %parametro che indica la semiampiezza del kernel monodimensionale utilizzato per estrarre il vicinato di ciascun pixel di bordo
stepCamp = sizeKern; %parametro che indica lo step di analisi della variabilità dei bordi, quanto più è vicino a 1, tanto più è una stima densa

 
%
%--------------inizializzazione strutture----------


%leggo immagine
I = imread(path_immagine);
Ior = I;
rig = size(I,1);
col = size(I,2);
contorni = cell(1,1);
regioni = cell(1,1);
direz_contorni = cell(1,1);
total_labeling = zeros(rig,col);
lastGlobLab = 1;
puntiConvDefect = [];
 

%applico filtro mediano
for i = 1:3
   I(:,:,i) = medfilt2(I(:,:,i),[10 10]); 
end

%estraggo comp blu
Ib = I(:,:,3);

%trasformo immagine da rgb a spazio lab
cform = makecform('srgb2lab');
lab_I = applycform(I,cform);
ab_I = double(lab_I(:,:,2:3));

%reshape dei pixel delle sole componenti a-b (dello spazio lab) come osservazioni in 2 dimensioni, per il clustering
obs = reshape ( ab_I , [rig * col 2]);
k = 3; %numero clusters ricercati
[labels , centroidi] = kmeans(obs,k,'distance','sqEuclidean','Replicates',3);
%reshape delle label nella forma dell'immagine 
labeled_I_1 = reshape(labels,rig,col);

 

%individuazioni delle regioni connesse (come subset delle regioni individuate dal k-means)
for i=1:k %per ciascuno dei cluster di colore (che non sono ancora regioni connesse)
   mask = (labeled_I_1 == i); 
   [label_comp_connesse, num_comp] = bwlabel(mask,8); %estraggo le regioni connesse per quel cluster di colore
   display(num_comp);
   for lab= 1:num_comp %per ciascuna regione connessa, estratta per quel cluster di colore
        mask2 = (label_comp_connesse == lab);
        
        %check sulla dimensione della regione rispetto all'immagine
        %(eliminiamo quelle troppo piccole)
        if nnz(mask2) < rig * col / 300
            continue;
        end
        
        %eliminiamo il mare
        media_blu =  mean(Ib(mask2 == 1));
        if media_blu > 100
            continue;
        end
        
        %erosione morfologica con 3 differenti se
        se = strel('disk',11);
        se2 = strel('line',11,300);
        se3 = strel('line',11,10);
        
        eroded = imerode(mask2,se2);
        eroded = imerode(eroded,se);
        eroded = imerode(eroded,se3); 
        
        %imshow(eroded);
        %waitforbuttonpress;
        
        %abbiamo eroso quella che era un'unica componente connessa...
        [dummylab, dummyNumC] = bwlabel(eroded,8);
        %...se risulta ancora abbastanza grande, allora la salviamo come
        %regione
        if nnz(eroded) < rig * col / 300 %|| dummyNumC > 1
            continue;
        end
        
        %salvo regione
        regioni{lastGlobLab} = mask2;
        
        %salvo etichetta su un'immagine finale che userò per mostrare tutte
        %le regioni, ciascuna con un colore (a seconda dell'etichetta)
        total_labeling (mask2) = lastGlobLab;
        
        
        
        %ESTRAZIONE CONTORNO REGIONE----------------------------
        %salvo contorno regione
        [startR, startC]= find(mask2 == 1);
        tcont = bwtraceboundary(mask2,[startR(1) startC(1)],'E');
        %salvo i punti
        cont = [];
        
        %elimino tutti quelli sul bordo immagine
        for p = 1:size(tcont,1)
           if tcont(p,1) == 1 || tcont(p,1) == rig || tcont(p,2) == 1 || tcont(p,2) == col
               continue;
           end
           cont = [cont; tcont(p,:)];
        end
        contorni{lastGlobLab}=cont;
      
        
        %OTTENIMENTO GRADIENTE DEI PUNTI DEL CONTORNO REGIONE-------------
        [grad, dirgrad] = imgradient(mask2,'prewitt');
        %in dir grad ci sono angoli direzione gradiente (alcuni negativi) e
        %a me interessa dare, per ogni punto in cui è calcolato il
        %gradiente (bordi doppi, non solo contorno) dare una misura di
        %quanto nell'intorno di quel punto la direzione del gradiente è
        %variabile
        %salvo per prima cosa i valori della direzione angolo gradiente nei
        %punti del contorno regione
        grad_cont = dirgrad( (cont(:,2)-1)*rig + cont(:,1) );  
        direz_contorni{lastGlobLab} = grad_cont;
         
        
        %ANALISI CONVESSITA' PUNTI CONTORNO
        if flag_analisi_conv == 1
            %voglio, per ogni punto del contorno regione, prendere il più vicino punto
            %del contorno del context hull, e se questo ha distanza > di una
            %costante per la diagonale del bounding box, lo considero un punto
            %di difetto di convessità
            %quindi per prima cosa devo calcolare la diagonale del bounding box
            STATS=regionprops(mask2,'BoundingBox');
            rect=round(STATS(1).BoundingBox);
            diagBB = sqrt(rect(3)*rect(3)+rect(4)*rect(4));
            %ora calcolo il contorno del convex hull
            convhull = bwconvhull(mask2);
            [startR, startC] = find(convhull == 1);
            tcont_convhull = bwtraceboundary(convhull,[startR(1) startC(1)],'E');
            %elimino punti sul bordo immagine
            cont_convhull = [];
            for p=1:size(tcont_convhull,1)
                if tcont_convhull(p,1) == 1 || tcont_convhull(p,1) == rig || tcont_convhull(p,2)==1 || tcont_convhull(p,2)==col
                    continue;
                end
                cont_convhull = [cont_convhull; [tcont_convhull(p,1) tcont_convhull(p,2)]];
            end
            
            %{
            imshow(mask2);
            hold on;
            plot(cont_convhull(:,2),cont_convhull(:,1),'.g');
            waitforbuttonpress;
            %}
            
            for k = 1:size(cont)
                a = [cont(k,1) cont(k,2)];
                d = ipdm(a,cont_convhull,'Subset','smallestfew','limit',1,'result','struct');

                if d.distance > 0.04 * diagBB %allora questo punto del contorno è un difetto di convessità
                   puntiConvDefect = [puntiConvDefect; [cont(k,1) cont(k,2)]  ]; 
                end
            end  
        end
 
        lastGlobLab = lastGlobLab + 1;
        
   end
end



 
visStep = 1.0 / (lastGlobLab-1);
visMat = double(visStep * total_labeling);
figure(1)
imshow(Ior);
figure(2)
imshow(visMat);
hold on;

%stampo i contorni
%calcolando la variabilità locale (per ciascun punto campionato del
%contorno) della direzione del gradiente, e a seconda di questa attribuisco
%un colore 


for i = 1:lastGlobLab-1
    %plotto tutto il contorno della regione
    %plot(contorni{i}(:,2), contorni{i}(:,1),'g');
    
    %muovo un kernel monodimensionale di direzione 2*sizeKern e calcolo la
    %dev standard dell'intorno di size, e questa è un amisura di quanto, in
    %quell'intorno, il gradiente cambia direzione
    
    for j = sizeKern+1 : stepCamp :size(contorni{i},1)-sizeKern
        standardDev = std( direz_contorni{i}(j-4:j+4)  );
       % display(standardDev);
        
       %nb: il plot del colore assegnato, va fatto per tutti i pixel del
       %contorno appartenenti almeno allo step
        if standardDev < 12
            for s= -stepCamp : stepCamp
                plot(contorni{i}(j+s,2), contorni{i}(j+s,1),'.g');
            end
            
        elseif standardDev < 50
             for s= -stepCamp : stepCamp
                plot(contorni{i}(j+s,2), contorni{i}(j+s,1),'.y');
             end
        else
             for s= -stepCamp : stepCamp
                plot(contorni{i}(j+s,2), contorni{i}(j+s,1),'.r');
             end
        end
        %waitforbuttonpress;
    end
    
    %waitforbuttonpress;
    
end

waitforbuttonpress;

if flag_analisi_conv == 1
    %%disegno sopra, anche i punti di maggiore convessità
    plot(puntiConvDefect(:,2),puntiConvDefect(:,1),'xc');
end





 


