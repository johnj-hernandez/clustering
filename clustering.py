import numpy as np 
import matplotlib.pyplot as plt
#el algoritmo fue creado para resolver un total de 2 features
#porlo que se usara el algoritmo KMeans de la libreria sklearn para evaluar la inertia
#para distintos numeros de clusters
from sklearn.cluster import KMeans as sk_Kmeans


class KMeans:
    def __init__(self, k,puntos):
        #numbers of clusters
        self.k=k
        #los datos 
        self.puntos=puntos
        #los k centroides seran en primera instancia los k primeros puntos
        self.centroids=np.copy(puntos[0:2])
        #ahora agregamos una etiqueta a cada punto indicando a que cluster pertenecen
        # en primera instancia todos pertenecen al cluster 1
        self.puntos=np.c_[puntos,np.zeros(len(puntos))]
        #luego hacemos que el punto 0 pertenezca a el cluster 0 representado por el mismo
        self.puntos[0][2]=0
        self.N=len(puntos)
        self.applyNIterations(10)


    def EuclidianDistance(self,x1,y1,x2,y2):
        #cada punto solo tiene 2 datos asi que solo tenemos x y y 
        return np.sqrt((x1-x2)**2  + (y1-y2)**2)
        
    #problema en la seleccion de clusteres
    def selectClusters(self):
        for i in range(self.N):
            d0=self.EuclidianDistance(self.puntos[i][0],self.puntos[i][1],self.centroids[0][0],self.centroids[0][1])
            d1=self.EuclidianDistance(self.puntos[i][0],self.puntos[i][1],self.centroids[1][0],self.centroids[1][1])
            if(d0<=d1):
                self.puntos[i][2]=0

            if(d1<d0):
                self.puntos[i][2]=1
            d0=0
            d1=0
    
    def updateCentroids(self):
        cx0=0
        cy0=0
        cx1=0
        cy1=0
        n0=0
        n1=0
        for i in range(self.N):
            if(self.puntos[i][2]==0):
                cx0+=self.puntos[i][0]
                cy0+=self.puntos[i][1]
                n0+=1
            if(self.puntos[i][2]==1):
                cx1+=self.puntos[i][0]
                cy1+=self.puntos[i][1]
                n1+=1
        self.centroids[0][0]=cx0/n0
        self.centroids[0][1]=cy0/n0
        self.centroids[1][0]=cx1/n1
        self.centroids[1][1]=cy1/n1


    def printData(self):
        print(self.puntos)
    
    def printCentroids(self):
        print("---------CENTROIDS-------")
        print(self.centroids)

    def applyNIterations(self,iterations):
        for i in range(iterations):
            #self.calculateInertia()
            self.selectClusters()
            self.updateCentroids()

    def plotData(self):
        x0=[]
        x1=[]
        y0=[]
        y1=[]
        plt.figure()   
        for i in range(self.N):
            if(self.puntos[i][2]==0):
                x0.append(self.puntos[i][0])
                y0.append(self.puntos[i][1])

            if(self.puntos[i][2]==1):
                x1.append(self.puntos[i][0])
                y1.append(self.puntos[i][1])
        plt.scatter(x0, y0, color='b', label='0')
        plt.scatter(x1, y1, color='r', label='1')
        plt.legend()
        plt.show()
    
    def calculateInertia(self):
        suma=0
        for i in range(self.N):
            centroid=int(self.puntos[i][2])
            distance=self.EuclidianDistance(self.puntos[i][0],self.puntos[i][1],self.centroids[centroid][0],self.centroids[centroid][1])
            suma+=distance**2
        print("Inertia: "+str(suma))
        return suma
        
def elbowPlot(data,maxKClusters):
    inertias=list()
    for i in range(1,maxKClusters+1):
        myCluster=sk_Kmeans(n_clusters=i)
        myCluster.fit(data)
        
        #myCluster=KMeans(i,data)
        
        inertias.append(myCluster.inertia_)
        
    plt.figure() 
    x=[i for i in range(1,maxKClusters+1)]
    y=[i for i in inertias]
    plt.plot(x,y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
    

if __name__ == "__main__":
    data=data=np.loadtxt("cluster_k2.txt",dtype=float,delimiter=";")
    clustering=KMeans(2,data)
    clustering.plotData()
    #generamos el grafico de codo para hasta 10 clusters
    #ahi se puede ver que despues de 2 clusters la variabilidad entre las inercias 
    #es muy minima
    elbowPlot(data,10)
