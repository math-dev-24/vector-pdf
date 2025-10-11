"""
Module pour créer des embeddings avec OpenAI.
Supporte le traitement par batch et le multithreading pour de meilleures performances.
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Charger les variables d'environnement
load_dotenv()


def create_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> List[List[float]]:
    """
    Crée des embeddings pour une liste de textes avec OpenAI.

    Args:
        texts: Liste de textes à vectoriser
        model: Modèle d'embedding OpenAI (text-embedding-3-small par défaut)
        batch_size: Taille des batchs pour l'API (max 100 pour OpenAI)

    Returns:
        Liste des vecteurs d'embeddings
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embeddings = []

    # Traiter par batch pour respecter les limites de l'API
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            input=batch,
            model=model
        )

        # Extraire les embeddings de la réponse
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def embed_chunks(
    chunks_data: List[Dict],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    verbose: bool = True,
    max_workers: int = 4
) -> List[Dict]:
    """
    Crée des embeddings pour des chunks avec leurs métadonnées.
    Utilise le multithreading pour traiter plusieurs batchs en parallèle.

    Args:
        chunks_data: Liste de dictionnaires avec 'content' et 'metadata'
        model: Modèle d'embedding OpenAI
        batch_size: Taille des batchs pour l'API
        verbose: Afficher la progression
        max_workers: Nombre maximum de threads (défaut: 4 pour respecter les limites de l'API)

    Returns:
        Liste de dictionnaires avec embeddings ajoutés
    """
    # Extraire les textes
    texts = [chunk['content'] for chunk in chunks_data]

    # Créer les batches
    num_batches = (len(texts) - 1) // batch_size + 1
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(texts))
        batches.append((batch_num, texts[start_idx:end_idx]))

    # Traiter les batches en parallèle
    embeddings_by_batch = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les jobs
        future_to_batch = {
            executor.submit(create_embeddings, batch_texts, model, batch_size): batch_num
            for batch_num, batch_texts in batches
        }

        # Traiter les résultats au fur et à mesure
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]

            try:
                batch_embeddings = future.result()
                embeddings_by_batch[batch_num] = batch_embeddings

                if verbose:
                    print(f"     Batch {batch_num + 1}/{num_batches}: ✅ {len(batch_embeddings)} embeddings créés")
            except Exception as e:
                if verbose:
                    print(f"     Batch {batch_num + 1}/{num_batches}: ❌ Erreur: {e}")
                raise

    # Reconstituer les embeddings dans l'ordre
    embeddings = []
    for batch_num in range(num_batches):
        embeddings.extend(embeddings_by_batch[batch_num])

    # Ajouter les embeddings aux chunks
    enriched_chunks = []
    for chunk, embedding in zip(chunks_data, embeddings):
        enriched_chunks.append({
            'content': chunk['content'],
            'metadata': chunk['metadata'],
            'embedding': embedding
        })

    if verbose:
        print(f"\n  ✅ {len(enriched_chunks)} embeddings créés")
        print(f"     Dimension des vecteurs: {len(embeddings[0])}")

    return enriched_chunks


def embed_all_files(
    results: List[Dict],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """
    Crée des embeddings pour tous les fichiers.

    Args:
        results: Résultats du chunking (liste de dicts avec 'chunks')
        model: Modèle d'embedding
        batch_size: Taille des batchs
        verbose: Afficher les logs de progression

    Returns:
        Liste enrichie avec embeddings
    """
    if verbose:
        print("\n🧠 Création des embeddings avec OpenAI...")
        print(f"   Modèle: {model}")

    total_chunks = sum(len(r['chunks']) for r in results)

    if verbose:
        print(f"   Total de chunks à vectoriser: {total_chunks}")
        print(f"   Taille des batchs: {batch_size}\n")

    # Collecter tous les chunks de tous les fichiers
    all_chunks = []
    for result in results:
        all_chunks.extend(result['chunks'])

    if verbose:
        print(f"  🔄 Vectorisation en cours...")
        num_batches = (len(all_chunks) - 1) // batch_size + 1
        print(f"     ({num_batches} batch(s) à traiter)\n")

    # Créer les embeddings
    enriched_chunks = embed_chunks(all_chunks, model=model, batch_size=batch_size, verbose=verbose)

    # Réorganiser par fichier
    enriched_results = []
    chunk_idx = 0

    for result in results:
        num_file_chunks = len(result['chunks'])
        file_chunks = enriched_chunks[chunk_idx:chunk_idx + num_file_chunks]

        enriched_results.append({
            'file_path': result['file_path'],
            'file_name': result['file_name'],
            'num_chunks': result['num_chunks'],
            'total_chars': result['total_chars'],
            'chunks': file_chunks
        })

        chunk_idx += num_file_chunks

    if verbose:
        print(f"\n✅ Vectorisation terminée: {total_chunks} embeddings créés")

    return enriched_results
