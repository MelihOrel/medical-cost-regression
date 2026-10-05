Sağlık sigortası masraflarını ne belirliyor? 1.338 kişilik bir veri setinde bunun cevabını aradım ve en ilginç sonuç beklemediğim yerden çıktı.

Sigara içmek tek başına masrafı ciddi şekilde artırıyor, bu sürpriz değil. Asıl çarpıcı olan şu: sigara içen biri aynı zamanda obez sınırını (vücut kitle indeksi 30) geçince masraf bir anda sıçrıyor. Kademeli bir artış değil, merdivende bir basamak çıkmak gibi. Sigara içen ve obez olan biri, aynı yaş ve kilodaki sigara içmeyen birine göre ortalama 33 bin dolar daha fazla ödüyor.

Bunu modele söylediğim anda tahmin hatası yüzde 26 düştü. Diğer bütün eklemeler toplamda 80 doları geçmedi.

Projeden aklımda kalan üç şey:

1. Grafiklere bakmadan model kurmayın. Yaş ve masraf grafiğinde üç paralel şerit vardı. Bu şeritlerin ne olduğunu çözmek, modelin tamamını çözmekti.

2. "Veri çarpık, logaritma alayım" refleksi burada işleri berbat etti. Model açıklama gücünün yarısından fazlasını kaybetti. Çünkü bu veride etkiler üst üste ekleniyor, çarpılmıyor. Dönüşümü varsaymak yerine test etmek gerekiyor.

3. Her veri setinin bir tavanı var. Sigara içmeyen 95 kişinin masrafı beklenenden çok yüksek ve verideki hiçbir bilgi bunu açıklamıyor. Muhtemelen kayıtta olmayan bir sağlık durumu. Bunu saklamak yerine raporda ayrı bir bölüm olarak yazdım.

Sonuçların ne kadar güvenilir olduğunu da üç farklı yöntemle kontrol ettim. Üçü de aynı şeyi söyledi.

Kod, grafikler ve tüm sonuçlar GitHub'da:
github.com/MelihOrel/medical-cost-regression

#veribilimi #istatistik #python #sağlıkverisi
