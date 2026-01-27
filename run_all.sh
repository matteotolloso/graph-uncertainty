

# python main.py --dataset chameleon --model vanilla --count 100    ;
# python main.py --dataset patents --model vanilla --count 100    ;
# python main.py --dataset arxiv --model vanilla --count 100    ;
# python main.py --dataset reddit2 --model vanilla --count 100    ;
# python main.py --dataset coauthor --model vanilla --count 100    ;
# python main.py --dataset squirrel --model vanilla --count 100    ;

# python main.py --dataset chameleon --model credal --count 30    ;
# python main.py --dataset patents --model credal --count 30   ;
# python main.py --dataset arxiv --model credal --count 30    ;
# python main.py --dataset reddit2 --model credal --count 30    ;
# python main.py --dataset coauthor --model credal --count 30    ;
# python main.py --dataset squirrel --model credal --count 30    ;

# python main.py --dataset chameleon --model credal_LJ --count 30    ;
# python main.py --dataset patents --model credal_LJ --count 30    ;
# python main.py --dataset arxiv --model credal_LJ --count 30    ;
# python main.py --dataset reddit2 --model credal_LJ --count 30    ;
# python main.py --dataset coauthor --model credal_LJ --count 30    ;
# python main.py --dataset squirrel --model credal_LJ --count 30    ;

# python main.py --dataset squirrel --model ensemble  ;
# python main.py --dataset arxiv --model ensemble  ;  
# python main.py --dataset coauthor --model ensemble  ;
# python main.py --dataset reddit2 --model ensemble  ;
# python main.py --dataset patents --model ensemble  ;
# python main.py --dataset chameleon --model ensemble  ;


# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model gebm  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model gebm  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model gebm ;


# python main.py --dataset squirrel --model frozen -c 30 ;
# python main.py --dataset arxiv --model frozen -c 30 ;
# python main.py --dataset coauthor --model frozen -c 30 ;
# python main.py --dataset reddit2 --model frozen -c 30 ;
# python main.py --dataset patents --model frozen -c 30 ;
# python main.py --dataset chameleon --model frozen -c 30 ;
# python main.py --dataset amazon_ratings --model frozen -c 30 ;
# python main.py --dataset cora --model frozen -c 30 ;
# python main.py --dataset roman_empire --model frozen -c 30 ;


# python main.py --dataset squirrel --model cagcn -c 30 ;
# python main.py --dataset arxiv --model cagcn -c 30 ;
# python main.py --dataset coauthor --model cagcn -c 30 ;
# python main.py --dataset reddit2 --model cagcn -c 30 ;
# python main.py --dataset patents --model cagcn -c 30 ;
# python main.py --dataset chameleon --model cagcn -c 30 ;
# python main.py --dataset amazon_ratings --model cagcn -c 30 ;
# python main.py --dataset cora --model cagcn -c 30 ;
# python main.py --dataset roman_empire --model cagcn -c 30 ;


# python main.py --dataset squirrel --model knn_LJ ;
# python main.py --dataset arxiv --model knn_LJ  ; 
# python main.py --dataset coauthor --model knn_LJ  ;
# python main.py --dataset reddit2 --model knn_LJ  ;  
# python main.py --dataset patents --model knn_LJ  ;  
# python main.py --dataset chameleon --model knn_LJ  ;
# python main.py --dataset amazon_ratings --model knn_LJ  ;
# python main.py --dataset cora --model knn_LJ  ;
# python main.py --dataset roman_empire --model knn_LJ  ;

# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model gnnsafe  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model gnnsafe  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model gnnsafe ;

# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model mahalanobis  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model mahalanobis  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model mahalanobis ;

# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model knn  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model knn  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model knn ;

# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset squirrel --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset arxiv --model energy  ;  
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset coauthor --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset reddit2 --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset patents --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset chameleon --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset amazon_ratings --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset cora --model energy  ;
# CUDA_VISIBLE_DEVICES=-1 python main.py --dataset roman_empire --model energy ;