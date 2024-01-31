# Journal

Journal for lab som inneholder alle notater og dokumentasjonen til koden vedlagt.

## Oppgaver
### 2.1 Forberedelsesoppgaver
1. Tegn et blokkskjema for hvordan en skal kople opp hele systemet med Raspberry Pi og 5 stk
AD-konvertere. Blokkskjemaet skal inneholde signalveier, klokkesignal, kommunikasjonsporter, og ledningsopplegg for spenningstilførsel til kretsene. I alle blokkene i skjemaet skal en
markere hvilke pinner som brukes. Vi anbefaler at dere bruker draw.io til å tegne blokkskjema.

2. Bli kjent med databladet for AD-konverteren vi bruker.
(a) Hvor mange klokkesykluser er det minste vi må bruke for å ta én punktprøve og overføre
dataene fra den tilbake til RPi?

(b) Gitt en driftspenning på 3.3 V (V_dd), hva blir AD-konverterens oppløsning? Oppgi
svaret i mV.

(c) Gitt en driftspenning på 3.3 V (V_dd) og jord på V_ss, hvor mye større spenning enn
V_dd, evt mindre enn V_ss, kan en pinne på AD-konverteren tåle?

3. I dette laboppsettet benytter vi oss av såkalt Direct Memory Access (DMA). Hva er den store fordelen med dette (evt. hvilket problem relatert til sampling av data med Linux løser dette
for oss)?

### 2.2 Laboppgaver
1. I labmanualen blir dere bedt om å bygge et enkelt lavpassfilter for å redusere støy på 3.3Vlinjen for spenningsforsyning til de analoge delene av systemet. Forklar hvordan filteret fungerer, beregn knekkfrekvens og tegn frekvensresponsen (Bode-plott) til filteret. Mål deretter frekvensresponsen vha Digilent.

2. Tegn opp kretsskjema, og kople opp 5 stk AD-konvertere av type MCP3201 slik at de kan
sample 5 kanaler samtidig (dvs at alle kanalene skal ha felles klokkesignal og chip-select, men separate data-utganger).

3. Demonstrer at dere kan sample et signal fra en signalkilde på alle 5 kanalene. Gjør en måling med adc_sampler på Pi-en og overfør måledataene til laptopen deres med en SFTP. NB:
husk at signalet må holde seg innenfor 0 – 3.3 V, så dere må legge på en DCoffset. Dataene skal vises som funksjon av tid. (Tips: Legg på ulike offset på hver kanal i SW
når dere plotter tidsserien slik at en kan skille de ulike kanalene fra hverandre.)

4. Skriv kode som lager et frekvensspektrum av dataene ved hjelp av FFT. Plott spekteret
av et sinussignal inn på AD-konverteren med kjent frekvens og verifiser at frekvensaksen til
plottet er korrekt. Forsøk å prosessere dataene med ulike vindusfunksjoner og zero-padding.
Studer spekteret for å se om dere får forventet SNR ut fra antall bit i ADC. Dukker det opp
noen uønskede (spuriøse) frekvenskomponenter i spekteret? Hva kan i tilfelle disse komme av?
Gjenta dette for ulike frekvenser til dere er sikre på at systemet oppfører seg som forventet.
(Kan med fordel kombineres med oppgave 2).