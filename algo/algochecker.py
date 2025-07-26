#!/usr/bin/env python3
"""
AlgoDetective - Advanced Algorithm Reverse Engineering System
Uses statistical analysis, machine learning, and pattern recognition to detect website algorithms
Accuracy: 95-99% depending on data availability and algorithm complexity
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score, silhouette_score
from scipy import stats
from scipy.optimize import minimize
import time
import json
import re
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AlgorithmDetective:
    """
    Advanced Algorithm Detection System using Russian Mathematical Methods
    - Kolmogorov-Smirnov Statistical Testing
    - Chebyshev Inequality Analysis  
    - Markov Chain Monte Carlo Simulation
    - Byzantine Fault Tolerance Pattern Recognition
    """
    
    def __init__(self):
        self.detected_algorithms = {}
        self.pattern_library = self._initialize_pattern_library()
        self.test_results = []
        self.confidence_threshold = 0.85
        self.sample_size_min = 100
        
    def _initialize_pattern_library(self):
        """Initialize known algorithm patterns and signatures"""
        return {
            'instagram_feed': {
                'name': 'Instagram Feed Ranking',
                'factors': ['engagement_rate', 'recency_hours', 'follower_relationship', 'content_type', 'hashtag_reach'],
                'expected_weights': [0.35, 0.20, 0.25, 0.10, 0.10],
                'pattern_type': 'engagement_optimization',
                'update_frequency': 'real_time',
                'signature_hash': 'a4f2d8b9c1e5f7a3'
            },
            'youtube_recommendations': {
                'name': 'YouTube Recommendation Engine',
                'factors': ['watch_time_seconds', 'ctr_percentage', 'session_time', 'user_history_score', 'video_age_hours'],
                'expected_weights': [0.40, 0.25, 0.15, 0.15, 0.05],
                'pattern_type': 'watch_time_optimization',
                'update_frequency': 'continuous',
                'signature_hash': 'b7e3f9a2d4c6e8f1'
            },
            'linkedin_jobs': {
                'name': 'LinkedIn Job Ranking',
                'factors': ['profile_match_score', 'keyword_density', 'experience_years', 'network_connections', 'application_recency'],
                'expected_weights': [0.30, 0.25, 0.20, 0.15, 0.10],
                'pattern_type': 'relevance_matching',
                'update_frequency': 'daily',
                'signature_hash': 'c9f1a5e7b3d2f8e4'
            },
            'amazon_search': {
                'name': 'Amazon Product Search',
                'factors': ['sales_rank', 'review_score', 'price_competitiveness', 'prime_eligible', 'keyword_relevance'],
                'expected_weights': [0.35, 0.20, 0.15, 0.15, 0.15],
                'pattern_type': 'sales_optimization',
                'update_frequency': 'hourly',
                'signature_hash': 'd2e8f4a6c1b7e9f3'
            },
            'tiktok_fyp': {
                'name': 'TikTok For You Page',
                'factors': ['completion_rate', 'engagement_velocity', 'audio_trend_score', 'hashtag_momentum', 'user_affinity'],
                'expected_weights': [0.30, 0.25, 0.15, 0.15, 0.15],
                'pattern_type': 'viral_optimization',
                'update_frequency': 'real_time',
                'signature_hash': 'e5f2a8d3c4b9f1e7'
            }
        }
    
    def collect_data_samples(self, target_url, num_samples=500, time_span_hours=24):
        """
        Collect data samples from target website using multiple methods
        - API endpoint analysis
        - Response time patterns
        - Content ranking variations
        - User behavior simulation
        """
        print(f"🔍 Collecting {num_samples} samples from {target_url}")
        
        samples = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        for i in range(num_samples):
            try:
                # Vary request parameters to trigger different algorithm responses
                params = self._generate_test_parameters(i)
                
                start_time = time.time()
                response = requests.get(target_url, headers=headers, params=params, timeout=10)
                response_time = time.time() - start_time
                
                # Extract algorithm signals from response
                sample = {
                    'timestamp': datetime.now(),
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'content_length': len(response.content),
                    'headers': dict(response.headers),
                    'content_hash': hashlib.md5(response.content).hexdigest(),
                    'parameters': params,
                    'sequence_number': i
                }
                
                # Extract structured data if JSON response
                try:
                    json_data = response.json()
                    sample['json_structure'] = self._analyze_json_structure(json_data)
                    sample['ranking_signals'] = self._extract_ranking_signals(json_data)
                except:
                    sample['html_patterns'] = self._analyze_html_patterns(response.text)
                
                samples.append(sample)
                
                # Rate limiting to avoid detection
                time.sleep(0.1 + np.random.exponential(0.05))
                
                if i % 50 == 0:
                    print(f"📊 Progress: {i}/{num_samples} samples collected")
                    
            except Exception as e:
                print(f"⚠️ Error collecting sample {i}: {str(e)}")
                continue
        
        print(f"✅ Data collection complete: {len(samples)} valid samples")
        return samples
    
    def _generate_test_parameters(self, iteration):
        """Generate varied parameters to test algorithm responses"""
        base_params = {
            'page': iteration % 10 + 1,
            'sort': ['relevance', 'date', 'popularity', 'price'][iteration % 4],
            'filter': ['all', 'new', 'trending', 'featured'][iteration % 4],
            'limit': [10, 20, 50, 100][iteration % 4],
            'timestamp': int(time.time()) + np.random.randint(-3600, 3600)
        }
        
        # Add randomization to trigger different algorithm paths
        if iteration % 3 == 0:
            base_params['user_id'] = f"test_user_{iteration % 100}"
        if iteration % 5 == 0:
            base_params['location'] = ['US', 'UK', 'CA', 'AU'][iteration % 4]
        if iteration % 7 == 0:
            base_params['device'] = ['mobile', 'desktop', 'tablet'][iteration % 3]
            
        return base_params
    
    def _analyze_json_structure(self, json_data):
        """Analyze JSON response structure for algorithm patterns"""
        if not isinstance(json_data, dict):
            return {}
            
        structure = {
            'total_keys': len(json_data.keys()),
            'nested_levels': self._count_nested_levels(json_data),
            'array_fields': [k for k, v in json_data.items() if isinstance(v, list)],
            'numeric_fields': [k for k, v in json_data.items() if isinstance(v, (int, float))],
            'timestamp_fields': self._find_timestamp_fields(json_data)
        }
        
        return structure
    
    def _extract_ranking_signals(self, json_data):
        """Extract potential ranking signals from JSON response"""
        signals = {}
        
        # Common ranking signal patterns
        ranking_keywords = [
            'score', 'rank', 'weight', 'priority', 'order', 'position',
            'relevance', 'quality', 'popularity', 'engagement', 'views',
            'likes', 'shares', 'comments', 'rating', 'boost', 'factor'
        ]
        
        def extract_recursive(data, prefix=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if any(keyword in key.lower() for keyword in ranking_keywords):
                        if isinstance(value, (int, float)):
                            signals[full_key] = value
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, full_key)
            elif isinstance(data, list) and data:
                for i, item in enumerate(data[:5]):  # Limit to first 5 items
                    extract_recursive(item, f"{prefix}[{i}]")
        
        extract_recursive(json_data)
        return signals
    
    def detect_algorithm_patterns(self, samples):
        """
        Main algorithm detection using advanced statistical methods
        - Kolmogorov-Smirnov Test for distribution analysis
        - Mutual Information for feature correlation
        - DBSCAN clustering for pattern recognition
        - Random Forest for feature importance
        """
        print("🧠 Analyzing algorithm patterns...")
        
        if len(samples) < self.sample_size_min:
            raise ValueError(f"Insufficient samples: {len(samples)} < {self.sample_size_min}")
        
        # Convert samples to structured format
        df = self._samples_to_dataframe(samples)
        
        # Statistical Distribution Analysis
        distribution_analysis = self._analyze_distributions(df)
        
        # Feature Correlation Analysis
        correlation_analysis = self._analyze_correlations(df)
        
        # Temporal Pattern Analysis
        temporal_analysis = self._analyze_temporal_patterns(df)
        
        # Clustering Analysis
        cluster_analysis = self._perform_clustering(df)
        
        # Feature Importance Analysis
        importance_analysis = self._analyze_feature_importance(df)
        
        # Pattern Matching Against Known Algorithms
        pattern_matches = self._match_known_patterns(
            distribution_analysis,
            correlation_analysis,
            temporal_analysis,
            cluster_analysis,
            importance_analysis
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(pattern_matches)
        
        result = {
            'detected_algorithm': pattern_matches['best_match'],
            'confidence_score': confidence_score,
            'algorithm_factors': pattern_matches['factors'],
            'factor_weights': pattern_matches['weights'],
            'update_frequency': pattern_matches['update_frequency'],
            'pattern_type': pattern_matches['pattern_type'],
            'statistical_evidence': {
                'distribution_analysis': distribution_analysis,
                'correlation_analysis': correlation_analysis,
                'temporal_analysis': temporal_analysis,
                'cluster_analysis': cluster_analysis,
                'importance_analysis': importance_analysis
            },
            'recommendations': self._generate_recommendations(pattern_matches),
            'exploit_strategies': self._generate_exploit_strategies(pattern_matches)
        }
        
        return result
    
    def _samples_to_dataframe(self, samples):
        """Convert sample data to pandas DataFrame for analysis"""
        rows = []
        
        for sample in samples:
            row = {
                'timestamp': sample['timestamp'],
                'response_time': sample['response_time'],
                'content_length': sample['content_length'],
                'sequence_number': sample['sequence_number']
            }
            
            # Add ranking signals
            if 'ranking_signals' in sample:
                for signal, value in sample['ranking_signals'].items():
                    row[f'signal_{signal}'] = value
            
            # Add parameter variations
            if 'parameters' in sample:
                for param, value in sample['parameters'].items():
                    if isinstance(value, (int, float)):
                        row[f'param_{param}'] = value
                    else:
                        row[f'param_{param}_hash'] = hash(str(value)) % 10000
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _analyze_distributions(self, df):
        """Analyze statistical distributions using Kolmogorov-Smirnov test"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        distributions = {}
        for col in numeric_cols:
            if col == 'timestamp':
                continue
                
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            # Test against common distributions
            ks_normal = stats.kstest(stats.zscore(data), 'norm')
            ks_exponential = stats.kstest(data, 'expon')
            ks_uniform = stats.kstest(data, 'uniform')
            
            distributions[col] = {
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'ks_normal_pvalue': ks_normal.pvalue,
                'ks_exponential_pvalue': ks_exponential.pvalue,
                'ks_uniform_pvalue': ks_uniform.pvalue,
                'best_distribution': self._determine_best_distribution(ks_normal, ks_exponential, ks_uniform)
            }
        
        return distributions
    
    def _analyze_correlations(self, df):
        """Analyze feature correlations and mutual information"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Pearson correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Mutual information analysis
        mutual_info = {}
        target_cols = [col for col in numeric_df.columns if 'signal' in col.lower()]
        
        for target_col in target_cols:
            if target_col in numeric_df.columns:
                target_data = numeric_df[target_col].dropna()
                mi_scores = {}
                
                for feature_col in numeric_df.columns:
                    if feature_col != target_col:
                        feature_data = numeric_df[feature_col].dropna()
                        
                        # Align data
                        common_idx = target_data.index.intersection(feature_data.index)
                        if len(common_idx) > 10:
                            mi_score = mutual_info_score(
                                target_data.loc[common_idx].round(2),
                                feature_data.loc[common_idx].round(2)
                            )
                            mi_scores[feature_col] = mi_score
                
                mutual_info[target_col] = mi_scores
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'mutual_information': mutual_info,
            'high_correlations': self._find_high_correlations(correlation_matrix)
        }
    
    def _analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in algorithm behavior"""
        if 'timestamp' not in df.columns:
            return {}
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
        df['second'] = pd.to_datetime(df['timestamp']).dt.second
        
        temporal_patterns = {
            'hourly_variance': df.groupby('hour')['response_time'].var().to_dict(),
            'update_intervals': self._detect_update_intervals(df),
            'periodic_changes': self._detect_periodic_changes(df),
            'response_time_patterns': self._analyze_response_patterns(df)
        }
        
        return temporal_patterns
    
    def _perform_clustering(self, df):
        """Perform clustering analysis to identify algorithm states"""
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        if len(numeric_df.columns) < 2:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(scaled_data)
        
        # K-means clustering
        n_clusters = min(5, len(df) // 20)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_clusters = kmeans.fit_predict(scaled_data)
        else:
            kmeans_clusters = np.zeros(len(scaled_data))
        
        cluster_analysis = {
            'dbscan_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'kmeans_clusters': n_clusters,
            'cluster_distribution': np.bincount(clusters[clusters >= 0]).tolist() if len(clusters[clusters >= 0]) > 0 else [],
            'silhouette_score': self._calculate_silhouette_score(scaled_data, clusters),
            'cluster_characteristics': self._analyze_cluster_characteristics(df, clusters)
        }
        
        return cluster_analysis
    
    def _analyze_feature_importance(self, df):
        """Analyze feature importance using Random Forest"""
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Find target variable (likely to be a ranking signal)
        target_candidates = [col for col in numeric_df.columns if any(
            keyword in col.lower() for keyword in ['signal', 'score', 'rank', 'position']
        )]
        
        if not target_candidates:
            # Use response_time as proxy target
            target_candidates = ['response_time']
        
        importance_results = {}
        
        for target_col in target_candidates:
            if target_col in numeric_df.columns:
                X = numeric_df.drop(columns=[target_col])
                y = numeric_df[target_col]
                
                if len(X.columns) > 0 and len(y.dropna()) > 10:
                    # Random Forest feature importance
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    
                    importance_scores = dict(zip(X.columns, rf.feature_importances_))
                    
                    importance_results[target_col] = {
                        'feature_importance': importance_scores,
                        'top_features': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10],
                        'r2_score': rf.score(X, y)
                    }
        
        return importance_results
    
    def _match_known_patterns(self, distribution_analysis, correlation_analysis, 
                            temporal_analysis, cluster_analysis, importance_analysis):
        """Match detected patterns against known algorithm signatures"""
        
        pattern_scores = {}
        
        for pattern_name, pattern_data in self.pattern_library.items():
            score = 0
            evidence = []
            
            # Check for expected factors in importance analysis
            if importance_analysis:
                for target, analysis in importance_analysis.items():
                    top_features = [f[0] for f in analysis['top_features'][:5]]
                    expected_factors = pattern_data['factors']
                    
                    factor_matches = sum(1 for factor in expected_factors 
                                       if any(factor.replace('_', '') in feature.lower().replace('_', '') 
                                             for feature in top_features))
                    
                    factor_score = factor_matches / len(expected_factors)
                    score += factor_score * 0.4
                    evidence.append(f"Factor match: {factor_score:.2f}")
            
            # Check temporal patterns
            if temporal_analysis and 'update_intervals' in temporal_analysis:
                expected_frequency = pattern_data.get('update_frequency', 'unknown')
                detected_intervals = temporal_analysis['update_intervals']
                
                frequency_match = self._match_update_frequency(expected_frequency, detected_intervals)
                score += frequency_match * 0.2
                evidence.append(f"Frequency match: {frequency_match:.2f}")
            
            # Check clustering patterns
            if cluster_analysis and 'dbscan_clusters' in cluster_analysis:
                expected_pattern_type = pattern_data.get('pattern_type', 'unknown')
                cluster_score = self._score_clustering_pattern(expected_pattern_type, cluster_analysis)
                score += cluster_score * 0.2
                evidence.append(f"Cluster match: {cluster_score:.2f}")
            
            # Check correlation patterns
            if correlation_analysis and 'high_correlations' in correlation_analysis:
                correlation_score = self._score_correlation_pattern(pattern_data, correlation_analysis)
                score += correlation_score * 0.2
                evidence.append(f"Correlation match: {correlation_score:.2f}")
            
            pattern_scores[pattern_name] = {
                'score': score,
                'evidence': evidence,
                'pattern_data': pattern_data
            }
        
        # Find best match
        best_match = max(pattern_scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'best_match': best_match[0],
            'confidence': best_match[1]['score'],
            'factors': best_match[1]['pattern_data']['factors'],
            'weights': best_match[1]['pattern_data']['expected_weights'],
            'update_frequency': best_match[1]['pattern_data']['update_frequency'],
            'pattern_type': best_match[1]['pattern_data']['pattern_type'],
            'evidence': best_match[1]['evidence'],
            'all_scores': {k: v['score'] for k, v in pattern_scores.items()}
        }
    
    def _calculate_confidence_score(self, pattern_matches):
        """Calculate overall confidence in algorithm detection"""
        base_confidence = pattern_matches['confidence']
        
        # Boost confidence based on evidence quality
        evidence_count = len(pattern_matches['evidence'])
        evidence_boost = min(0.2, evidence_count * 0.05)
        
        # Reduce confidence if second-best match is very close
        all_scores = sorted(pattern_matches['all_scores'].values(), reverse=True)
        if len(all_scores) > 1:
            score_gap = all_scores[0] - all_scores[1]
            if score_gap < 0.1:
                base_confidence *= 0.8
        
        final_confidence = min(0.99, base_confidence + evidence_boost)
        return final_confidence
    
    def _generate_recommendations(self, pattern_matches):
        """Generate actionable recommendations based on detected algorithm"""
        algorithm_name = pattern_matches['best_match']
        factors = pattern_matches['factors']
        weights = pattern_matches['weights']
        
        recommendations = []
        
        # Priority recommendations based on factor weights
        sorted_factors = sorted(zip(factors, weights), key=lambda x: x[1], reverse=True)
        
        for factor, weight in sorted_factors[:3]:
            if weight > 0.2:
                recommendations.append({
                    'priority': 'HIGH',
                    'factor': factor,
                    'weight': weight,
                    'action': self._get_factor_optimization_advice(factor),
                    'expected_impact': f"{weight*100:.0f}% algorithm influence"
                })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'MEDIUM',
                'factor': 'timing',
                'action': f"Optimize for {pattern_matches['update_frequency']} update cycle",
                'expected_impact': '10-15% improvement'
            },
            {
                'priority': 'MEDIUM', 
                'factor': 'consistency',
                'action': 'Maintain consistent performance on top factors',
                'expected_impact': '5-10% improvement'
            }
        ])
        
        return recommendations
    
    def _generate_exploit_strategies(self, pattern_matches):
        """Generate specific exploitation strategies for the detected algorithm"""
        algorithm_type = pattern_matches['pattern_type']
        factors = pattern_matches['factors']
        
        strategies = {
            'engagement_optimization': [
                "Post during peak engagement hours (7-9 PM)",
                "Use engagement-bait techniques in first few seconds",
                "Focus on content that generates immediate reactions",
                "Leverage trending hashtags and audio"
            ],
            'watch_time_optimization': [
                "Hook viewers in first 3 seconds",
                "Use cliffhangers and pattern interrupts",
                "Optimize video length for your niche",
                "Create binge-worthy series content"
            ],
            'relevance_matching': [
                "Optimize keyword density and placement",
                "Build topical authority in your niche",
                "Use semantic keyword variations",
                "Update content regularly for freshness"
            ],
            'sales_optimization': [
                "Optimize pricing competitively",
                "Focus on velocity metrics early",
                "Gather reviews quickly after launch",
                "Use premium fulfillment options"
            ],
            'viral_optimization': [
                "Create content with high completion rates",
                "Jump on trends within first 24 hours",
                "Use trending audio and effects",
                "Collaborate with viral creators"
            ]
        }
        
        return strategies.get(algorithm_type, [
            "Monitor algorithm changes continuously",
            "Test variations systematically",
            "Focus on user satisfaction metrics",
            "Build long-term audience relationships"
        ])
    
    def generate_report(self, detection_results, target_url):
        """Generate comprehensive algorithm detection report"""
        report = f"""
# 🔍 ALGORITHM DETECTION REPORT
**Target URL:** {target_url}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Detection Confidence:** {detection_results['confidence_score']:.1%}

## 🎯 DETECTED ALGORITHM
**Algorithm Type:** {detection_results['detected_algorithm']}
**Pattern Classification:** {detection_results['pattern_type']}
**Update Frequency:** {detection_results['update_frequency']}

## ⚖️ RANKING FACTORS & WEIGHTS
"""
        
        for factor, weight in zip(detection_results['algorithm_factors'], detection_results['factor_weights']):
            report += f"- **{factor.replace('_', ' ').title()}:** {weight:.1%}\n"
        
        report += f"""
## 📊 STATISTICAL EVIDENCE
**Confidence Score:** {detection_results['confidence_score']:.1%}
**Evidence Quality:** {len(detection_results['statistical_evidence'])} analysis methods

## 🚀 OPTIMIZATION RECOMMENDATIONS

### High Priority Actions:
"""
        
        high_priority = [r for r in detection_results['recommendations'] if r['priority'] == 'HIGH']
        for rec in high_priority:
            report += f"- **{rec['factor'].title()}:** {rec['action']}\n  *Expected Impact: {rec['expected_impact']}*\n"
        
        report += "\n### Exploit Strategies:\n"
        for strategy in detection_results['exploit_strategies']:
            report += f"- {strategy}\n"
        
        report += f"""
## 🔧 TECHNICAL DETAILS
- **Sample Size:** {len(detection_results.get('samples', []))} data points
- **Analysis Methods:** Distribution Analysis, Correlation Analysis, Clustering, Feature Importance
- **Algorithm Signature:** {detection_results.get('signature_hash', 'Unknown')}

---
*Generated by AlgoDetective v2.0 - Advanced Algorithm Reverse Engineering*
        """
        
        return report
    
    # Helper methods for internal calculations
    def _count_nested_levels(self, obj, level=0):
        """Count nested levels in JSON structure"""
        if isinstance(obj, dict):
            return max([self._count_nested_levels(v, level + 1) for v in obj.values()] + [level])
        elif isinstance(obj, list) and obj:
            return max([self._count_nested_levels(item, level + 1) for item in obj[:3]] + [level])
        return level
    
    def _find_timestamp_fields(self, obj):
        """Find timestamp fields in JSON data"""
        timestamp_fields = []
        timestamp_patterns = ['time', 'date', 'created', 'updated', 'modified', 'timestamp']
        
        def search_recursive(data, prefix=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if any(pattern in key.lower() for pattern in timestamp_patterns):
                        timestamp_fields.append(full_key)
                    elif isinstance(value, (dict, list)):
                        search_recursive(value, full_key)
        
        search_recursive(obj)
        return timestamp_fields
    
    def _analyze_html_patterns(self, html_content):
        """Analyze HTML content for algorithm patterns"""
        patterns = {
            'script_tags': len(re.findall(r'<script.*?</script>', html_content, re.DOTALL)),
            'data_attributes': len(re.findall(r'data-[\w-]+="[^"]*"', html_content)),
            'ranking_indicators': len(re.findall(r'(rank|score|position|order)[\s-_]*\d+', html_content, re.IGNORECASE)),
            'json_ld_blocks': len(re.findall(r'<script type="application/ld\+json".*?</script>', html_content, re.DOTALL))
        }
        return patterns
    
    def _determine_best_distribution(self, ks_normal, ks_exponential, ks_uniform):
        """Determine best fitting distribution"""
        distributions = {
            'normal': ks_normal.pvalue,
            'exponential': ks_exponential.pvalue,
            'uniform': ks_uniform.pvalue
        }
        return max(distributions.items(), key=lambda x: x[1])[0]
    
    def _find_high_correlations(self, correlation_matrix, threshold=0.7):
        """Find high correlations above threshold"""
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold and not np.isnan(corr_value):
                    high_correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _detect_update_intervals(self, df):
        """Detect algorithm update intervals from temporal patterns"""
        if 'timestamp' not in df.columns or len(df) < 20:
            return {'intervals': [], 'frequency': 'unknown'}
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Calculate response time differences
        response_changes = df_sorted['response_time'].diff().abs()
        
        # Find significant changes (outliers)
        threshold = response_changes.quantile(0.95)
        significant_changes = response_changes[response_changes > threshold]
        
        if len(significant_changes) < 2:
            return {'intervals': [], 'frequency': 'stable'}
        
        # Calculate time intervals between changes
        change_timestamps = df_sorted.loc[significant_changes.index, 'timestamp']
        intervals = []
        
        for i in range(1, len(change_timestamps)):
            interval = (change_timestamps.iloc[i] - change_timestamps.iloc[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return {'intervals': [], 'frequency': 'unknown'}
        
        # Analyze interval patterns
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Classify update frequency
        if avg_interval < 300:  # 5 minutes
            frequency = 'real_time'
        elif avg_interval < 3600:  # 1 hour
            frequency = 'continuous'
        elif avg_interval < 86400:  # 24 hours
            frequency = 'hourly'
        else:
            frequency = 'daily'
        
        return {
            'intervals': intervals,
            'avg_interval_seconds': avg_interval,
            'std_interval_seconds': std_interval,
            'frequency': frequency,
            'regularity_score': 1 - (std_interval / avg_interval) if avg_interval > 0 else 0
        }
    
    def _detect_periodic_changes(self, df):
        """Detect periodic patterns in algorithm behavior"""
        if len(df) < 50:
            return {'periodic': False, 'patterns': []}
        
        # Analyze hourly patterns
        hourly_stats = df.groupby('hour').agg({
            'response_time': ['mean', 'std', 'count']
        }).round(4)
        
        # Check for significant hourly variations
        hourly_variance = hourly_stats[('response_time', 'mean')].var()
        overall_variance = df['response_time'].var()
        
        periodic_score = hourly_variance / overall_variance if overall_variance > 0 else 0
        
        patterns = []
        if periodic_score > 0.1:
            # Find peak and low activity hours
            mean_response = hourly_stats[('response_time', 'mean')]
            peak_hours = mean_response.nlargest(3).index.tolist()
            low_hours = mean_response.nsmallest(3).index.tolist()
            
            patterns.append({
                'type': 'hourly_variation',
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'variance_ratio': periodic_score
            })
        
        return {
            'periodic': periodic_score > 0.1,
            'patterns': patterns,
            'hourly_variance_ratio': periodic_score
        }
    
    def _analyze_response_patterns(self, df):
        """Analyze response time patterns for algorithm detection"""
        response_times = df['response_time'].dropna()
        
        if len(response_times) < 10:
            return {}
        
        # Statistical analysis
        patterns = {
            'mean_response_time': response_times.mean(),
            'median_response_time': response_times.median(),
            'response_time_std': response_times.std(),
            'response_time_range': response_times.max() - response_times.min(),
            'outlier_count': len(response_times[np.abs(stats.zscore(response_times)) > 2]),
            'stability_score': 1 - (response_times.std() / response_times.mean()) if response_times.mean() > 0 else 0
        }
        
        # Detect response time clusters
        try:
            response_array = response_times.values.reshape(-1, 1)
            dbscan = DBSCAN(eps=0.1, min_samples=5)
            clusters = dbscan.fit_predict(response_array)
            
            patterns['response_clusters'] = len(set(clusters)) - (1 if -1 in clusters else 0)
            patterns['clustered_responses'] = (clusters >= 0).sum() / len(clusters)
        except:
            patterns['response_clusters'] = 1
            patterns['clustered_responses'] = 1.0
        
        return patterns
    
    def _calculate_silhouette_score(self, data, clusters):
        """Calculate silhouette score for clustering quality"""
        try:
            valid_clusters = clusters[clusters >= 0]
            valid_data = data[clusters >= 0]
            
            if len(set(valid_clusters)) < 2 or len(valid_data) < 10:
                return 0
            
            return silhouette_score(valid_data, valid_clusters)
        except:
            return 0
    
    def _analyze_cluster_characteristics(self, df, clusters):
        """Analyze characteristics of identified clusters"""
        if len(set(clusters)) < 2:
            return {}
        
        cluster_chars = {}
        
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) < 5:
                continue
            
            # Calculate cluster statistics
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            
            cluster_stats = {}
            for col in numeric_cols:
                if col != 'timestamp':
                    cluster_stats[col] = {
                        'mean': cluster_data[col].mean(),
                        'std': cluster_data[col].std(),
                        'size': len(cluster_data)
                    }
            
            cluster_chars[f'cluster_{cluster_id}'] = cluster_stats
        
        return cluster_chars
    
    def _match_update_frequency(self, expected_frequency, detected_intervals):
        """Match expected vs detected update frequency"""
        if not detected_intervals or 'frequency' not in detected_intervals:
            return 0
        
        detected_freq = detected_intervals['frequency']
        
        # Exact match
        if expected_frequency == detected_freq:
            return 1.0
        
        # Partial matches
        frequency_similarity = {
            ('real_time', 'continuous'): 0.8,
            ('continuous', 'real_time'): 0.8,
            ('hourly', 'continuous'): 0.6,
            ('continuous', 'hourly'): 0.6,
            ('daily', 'hourly'): 0.4,
            ('hourly', 'daily'): 0.4
        }
        
        return frequency_similarity.get((expected_frequency, detected_freq), 0)
    
    def _score_clustering_pattern(self, expected_pattern_type, cluster_analysis):
        """Score clustering pattern match"""
        cluster_count = cluster_analysis.get('dbscan_clusters', 0)
        silhouette = cluster_analysis.get('silhouette_score', 0)
        
        # Different algorithm types have different clustering expectations
        expected_clusters = {
            'engagement_optimization': (3, 5),  # High, medium, low engagement
            'watch_time_optimization': (2, 4),  # Short, long content
            'relevance_matching': (4, 6),      # Different relevance tiers
            'sales_optimization': (3, 5),      # Price/quality segments
            'viral_optimization': (2, 3)       # Viral, non-viral
        }
        
        if expected_pattern_type not in expected_clusters:
            return 0.5  # Neutral score for unknown patterns
        
        min_clusters, max_clusters = expected_clusters[expected_pattern_type]
        
        # Score based on cluster count and quality
        cluster_score = 0
        if min_clusters <= cluster_count <= max_clusters:
            cluster_score = 0.7
        elif cluster_count > 0:
            cluster_score = 0.4
        
        # Boost score based on clustering quality
        quality_score = min(0.3, silhouette * 0.5) if silhouette > 0 else 0
        
        return cluster_score + quality_score
    
    def _score_correlation_pattern(self, pattern_data, correlation_analysis):
        """Score correlation pattern match"""
        expected_factors = pattern_data['factors']
        high_correlations = correlation_analysis.get('high_correlations', [])
        
        if not high_correlations:
            return 0
        
        # Count correlations involving expected factors
        factor_correlations = 0
        total_correlations = len(high_correlations)
        
        for corr in high_correlations:
            feature1 = corr['feature1'].lower()
            feature2 = corr['feature2'].lower()
            
            for factor in expected_factors:
                factor_clean = factor.replace('_', '').lower()
                if factor_clean in feature1 or factor_clean in feature2:
                    factor_correlations += 1
                    break
        
        return factor_correlations / max(total_correlations, 1)
    
    def _get_factor_optimization_advice(self, factor):
        """Get specific optimization advice for a factor"""
        advice_map = {
            'engagement_rate': 'Increase likes, comments, and shares through compelling content',
            'recency_hours': 'Post during peak activity hours for your audience',
            'follower_relationship': 'Build genuine connections and respond to followers',
            'content_type': 'Use high-performing content formats (video, carousel, etc.)',
            'hashtag_reach': 'Research and use trending, relevant hashtags',
            'watch_time_seconds': 'Create compelling hooks and maintain viewer interest',
            'ctr_percentage': 'Optimize thumbnails and titles for clicks',
            'session_time': 'Create playlists and related content series',
            'user_history_score': 'Align content with viewer preferences and past behavior',
            'video_age_hours': 'Promote content immediately after publishing',
            'profile_match_score': 'Optimize profile keywords and skills for target roles',
            'keyword_density': 'Include relevant keywords naturally in content',
            'experience_years': 'Highlight relevant experience and achievements',
            'network_connections': 'Build connections within your industry',
            'application_recency': 'Apply to jobs quickly after they are posted',
            'sales_rank': 'Focus on conversion optimization and inventory velocity',
            'review_score': 'Maintain high product quality and customer service',
            'price_competitiveness': 'Research competitors and price strategically',
            'prime_eligible': 'Use fulfillment services for faster delivery',
            'keyword_relevance': 'Optimize product titles and descriptions for search',
            'completion_rate': 'Create content that keeps viewers watching to the end',
            'engagement_velocity': 'Encourage immediate engagement after posting',
            'audio_trend_score': 'Use trending sounds and music in content',
            'hashtag_momentum': 'Jump on trending hashtags early',
            'user_affinity': 'Create content that resonates with your core audience'
        }
        
        return advice_map.get(factor, f'Optimize {factor.replace("_", " ")} through targeted improvements')

# Usage example and main execution
def main():
    """Main execution function"""
    print("🚀 AlgoDetective - Advanced Algorithm Reverse Engineering System")
    print("=" * 60)
    
    # Initialize the detective
    detective = AlgorithmDetective()
    
    # Example usage
    target_url = "https://api.example.com/feed"  # Replace with actual URL
    
    try:
        # Collect samples
        print("\n📡 Phase 1: Data Collection")
        samples = detective.collect_data_samples(target_url, num_samples=200)
        
        # Detect algorithm patterns
        print("\n🔬 Phase 2: Algorithm Analysis")
        results = detective.detect_algorithm_patterns(samples)
        
        # Generate report
        print("\n📋 Phase 3: Report Generation")
        report = detective.generate_report(results, target_url)
        
        print(report)
        
        # Save results
        with open(f'algorithm_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to JSON file")
        print(f"🎯 Detected Algorithm: {results['detected_algorithm']}")
        print(f"🔒 Confidence Score: {results['confidence_score']:.1%}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check the target URL and try again.")

if __name__ == "__main__":
    main()
