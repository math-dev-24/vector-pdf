"""
Module pour créer des embeddings avec OpenAI.
Supporte le traitement par batch et le multithreading pour de meilleures performances.
Utilise le cache et les singletons pour optimiser les performances.
"""

import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core import (
    OpenAIClient,
    get_logger,
    PipelineError,
    ErrorType,
    settings,
    retry_with_backoff
)
from src.core.cache import get_embedding_cache

logger = get_logger(__name__)


@retry_with_backoff(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=30.0,
    exceptions=(Exception,)
)
def create_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: Optional[int] = None
) -> tuple[List[List[float]], int]:
    """
    Crée des embeddings pour une liste de textes avec OpenAI.
    Utilise le cache pour éviter les appels API redondants.
    Retourne aussi la dimension détectée.

    Args:
        texts: Liste de textes à vectoriser
        model: Modèle d'embedding OpenAI (défaut: settings.embedding_model)
        batch_size: Taille des batchs pour l'API (défaut: settings.embedding_batch_size)

    Returns:
        Tuple (liste des vecteurs d'embeddings, dimension détectée)
    """
    model = model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size
    
    client = OpenAIClient().client
    cache = get_embedding_cache()
    
    embeddings = []
    texts_to_fetch = []
    indices_to_fetch = []
    cached_embeddings = {}
    detected_dimension = None

    # Vérifier le cache pour chaque texte
    for idx, text in enumerate(texts):
        cached = cache.get(text, model)
        if cached:
            cached_embeddings[idx] = cached
            # Détecter la dimension depuis le cache
            if detected_dimension is None:
                detected_dimension = len(cached)
            logger.debug(f"Cache hit pour texte {idx}")
        else:
            texts_to_fetch.append((idx, text))
            indices_to_fetch.append(idx)

    # Récupérer les embeddings depuis le cache
    for idx, embedding in cached_embeddings.items():
        embeddings.append((idx, embedding))

    # Si tous les textes sont en cache, retourner dans l'ordre
    if not texts_to_fetch:
        sorted_embeddings = [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
        return sorted_embeddings, detected_dimension or 1536

    # Traiter par batch pour respecter les limites de l'API
    fetched_texts = [text for _, text in texts_to_fetch]
    
    try:
        for i in range(0, len(fetched_texts), batch_size):
            batch = fetched_texts[i:i + batch_size]
            batch_indices = indices_to_fetch[i:i + batch_size]

            response = client.embeddings.create(
                input=batch,
                model=model
            )

            # Détecter la dimension depuis la première réponse
            if detected_dimension is None and response.data:
                detected_dimension = len(response.data[0].embedding)
                logger.debug(f"Dimension détectée: {detected_dimension}")

            # Extraire les embeddings de la réponse et mettre en cache
            for idx, item in zip(batch_indices, response.data):
                embedding = item.embedding
                embeddings.append((idx, embedding))
                # Mettre en cache
                cache.set(texts[idx], model, embedding)
                logger.debug(f"Embedding créé et mis en cache pour texte {idx}")

    except Exception as e:
        raise PipelineError(
            ErrorType.EMBEDDING,
            f"Erreur lors de la création des embeddings: {e}",
            original_error=e
        )

    # Retourner dans l'ordre original avec la dimension
    sorted_embeddings = [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
    return sorted_embeddings, detected_dimension or 1536


async def create_embeddings_async(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: Optional[int] = None
) -> tuple[List[List[float]], int]:
    """
    Version asynchrone de create_embeddings.
    Utilise AsyncOpenAI via le singleton OpenAIClient.

    Args:
        texts: Liste de textes à vectoriser
        model: Modèle d'embedding
        batch_size: Taille des batchs

    Returns:
        Tuple (liste des vecteurs d'embeddings, dimension détectée)
    """
    model = model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size

    client = OpenAIClient().async_client
    cache = get_embedding_cache()

    embeddings = []
    texts_to_fetch = []
    indices_to_fetch = []
    cached_embeddings = {}
    detected_dimension = None

    # Vérifier le cache pour chaque texte
    for idx, text in enumerate(texts):
        cached = cache.get(text, model)
        if cached:
            cached_embeddings[idx] = cached
            if detected_dimension is None:
                detected_dimension = len(cached)
            logger.debug(f"Cache hit pour texte {idx}")
        else:
            texts_to_fetch.append((idx, text))
            indices_to_fetch.append(idx)

    # Récupérer les embeddings depuis le cache
    for idx, embedding in cached_embeddings.items():
        embeddings.append((idx, embedding))

    # Si tous les textes sont en cache, retourner dans l'ordre
    if not texts_to_fetch:
        sorted_embeddings = [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
        return sorted_embeddings, detected_dimension or 1536

    fetched_texts = [text for _, text in texts_to_fetch]

    async def _fetch_batch(batch: List[str], batch_indices: List[int]) -> None:
        nonlocal detected_dimension
        response = await client.embeddings.create(
            input=batch,
            model=model
        )
        if detected_dimension is None and response.data:
            detected_dimension = len(response.data[0].embedding)
            logger.debug(f"Dimension détectée (async): {detected_dimension}")

        for idx, item in zip(batch_indices, response.data):
            embedding = item.embedding
            embeddings.append((idx, embedding))
            cache.set(texts[idx], model, embedding)

    # Lancer les batchs en parallèle
    tasks = []
    for i in range(0, len(fetched_texts), batch_size):
        batch = fetched_texts[i:i + batch_size]
        batch_indices = indices_to_fetch[i:i + batch_size]
        tasks.append(_fetch_batch(batch, batch_indices))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        raise PipelineError(
            ErrorType.EMBEDDING,
            f"Erreur lors de la création des embeddings (async): {e}",
            original_error=e
        )

    sorted_embeddings = [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]
    return sorted_embeddings, detected_dimension or 1536


def embed_chunks(
    chunks_data: List[Dict],
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Crée des embeddings pour des chunks avec leurs métadonnées.
    Utilise le multithreading pour traiter plusieurs batchs en parallèle.
    Utilise le cache pour optimiser les performances.

    Args:
        chunks_data: Liste de dictionnaires avec 'content' et 'metadata'
        model: Modèle d'embedding OpenAI (défaut: settings.embedding_model)
        batch_size: Taille des batchs pour l'API (défaut: settings.embedding_batch_size)
        verbose: Afficher la progression
        max_workers: Nombre maximum de threads (défaut: auto-configuré dynamiquement)

    Returns:
        Liste de dictionnaires avec embeddings ajoutés
    """
    model = model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size
    
    # Extraire les textes
    texts = [chunk['content'] for chunk in chunks_data]

    # Créer les batches
    num_batches = (len(texts) - 1) // batch_size + 1
    
    # Configuration dynamique des workers
    if max_workers is None:
        # Limiter à 4 pour respecter les limites API, mais adapter selon les batches
        max_workers = min(
            settings.max_workers or 4,
            num_batches,  # Ne pas créer plus de workers que de batches
            4  # Limite de sécurité pour l'API
        )
    
    if verbose:
        logger.info(f"Configuration: {num_batches} batch(s), {max_workers} worker(s)")

    # Créer les batches
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(texts))
        batches.append((batch_num, texts[start_idx:end_idx]))

    # Traiter les batches en parallèle
    embeddings_by_batch = {}
    detected_dimension = None

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
                batch_embeddings, dimension = future.result()
                embeddings_by_batch[batch_num] = batch_embeddings
                
                # Capturer la dimension détectée
                if detected_dimension is None:
                    detected_dimension = dimension

                if verbose:
                    logger.info(f"Batch {batch_num + 1}/{num_batches}: {len(batch_embeddings)} embeddings créés")
            except Exception as e:
                if verbose:
                    logger.error(f"Batch {batch_num + 1}/{num_batches}: Erreur: {e}")
                raise PipelineError(
                    ErrorType.EMBEDDING,
                    f"Erreur lors du traitement du batch {batch_num + 1}: {e}",
                    original_error=e
                )

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
        cache_stats = get_embedding_cache().get_stats()
        dimension = detected_dimension or (len(embeddings[0]) if embeddings else 0)
        logger.info(f"{len(enriched_chunks)} embeddings créés (dimension: {dimension})")
        logger.info(f"Cache: {cache_stats['total_cached']} embeddings en cache ({cache_stats['cache_size_mb']} MB)")

    return enriched_chunks


async def embed_chunks_async(
    chunks_data: List[Dict],
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Version asynchrone de embed_chunks.
    Utilise asyncio pour traiter les batchs en parallèle.

    Args:
        chunks_data: Liste de chunks
        model: Modèle d'embedding
        batch_size: Taille des batchs
        verbose: Afficher la progression

    Returns:
        Liste de dictionnaires avec embeddings ajoutés
    """
    model = model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size

    texts = [chunk['content'] for chunk in chunks_data]
    num_batches = (len(texts) - 1) // batch_size + 1

    # Créer les batchs
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(texts))
        batches.append((batch_num, texts[start_idx:end_idx]))

    embeddings_by_batch = {}
    detected_dimension = None

    async def _process_batch(batch_num: int, batch_texts: List[str]) -> None:
        nonlocal detected_dimension
        batch_embeddings, dimension = await create_embeddings_async(
            batch_texts,
            model,
            batch_size
        )
        embeddings_by_batch[batch_num] = batch_embeddings
        if detected_dimension is None:
            detected_dimension = dimension
        if verbose:
            logger.info(f"Batch async {batch_num + 1}/{num_batches}: {len(batch_embeddings)} embeddings créés")

    tasks = [_process_batch(batch_num, batch_texts) for batch_num, batch_texts in batches]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        if verbose:
            logger.error(f"Erreur async embedding: {e}")
        raise PipelineError(
            ErrorType.EMBEDDING,
            f"Erreur lors du traitement async: {e}",
            original_error=e
        )

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
        cache_stats = get_embedding_cache().get_stats()
        dimension = detected_dimension or (len(embeddings[0]) if embeddings else 0)
        logger.info(f"{len(enriched_chunks)} embeddings créés (dimension: {dimension}) [async]")
        logger.info(f"Cache: {cache_stats['total_cached']} embeddings en cache ({cache_stats['cache_size_mb']} MB)")

    return enriched_chunks


def embed_all_files(
    results: List[Dict],
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
    use_smart_batching: bool = None
) -> List[Dict]:
    """
    Crée des embeddings pour tous les fichiers.
    Utilise le cache et le smart batching pour optimiser les performances.

    Args:
        results: Résultats du chunking (liste de dicts avec 'chunks')
        model: Modèle d'embedding (défaut: settings.embedding_model)
        batch_size: Taille des batchs (défaut: settings.embedding_batch_size)
        verbose: Afficher les logs de progression
        use_smart_batching: Utiliser smart batching (défaut: settings.smart_batching)

    Returns:
        Liste enrichie avec embeddings
    """
    model = model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size
    use_smart_batching = use_smart_batching if use_smart_batching is not None else settings.smart_batching
    use_async = settings.enable_async
    
    if verbose:
        logger.info("Création des embeddings avec OpenAI...")
        logger.info(f"Modèle: {model}")

    total_chunks = sum(len(r['chunks']) for r in results)

    if verbose:
        logger.info(f"Total de chunks à vectoriser: {total_chunks}")

    # Collecter tous les chunks de tous les fichiers
    all_chunks = []
    for result in results:
        all_chunks.extend(result['chunks'])

    # Utiliser smart batching si activé
    if use_smart_batching:
        from src.vectorization.smart_batching import SmartBatcher
        batcher = SmartBatcher()
        smart_batches = batcher.create_smart_batches(all_chunks)
        
        if verbose:
            stats = batcher.get_batch_stats(smart_batches)
            logger.info(f"Smart batching: {stats['num_batches']} batch(s) créé(s)")
            logger.info(f"  Moyenne: {stats['avg_batch_size']:.1f} chunks/batch, "
                       f"{stats['avg_tokens_per_batch']:.0f} tokens/batch")
        
        # Traiter chaque smart batch
        enriched_chunks = []
        for i, smart_batch in enumerate(smart_batches, 1):
            if verbose:
                logger.info(f"Traitement du batch intelligent {i}/{len(smart_batches)} "
                           f"({len(smart_batch)} chunks)")
            
            if use_async:
                batch_enriched = asyncio.run(
                    embed_chunks_async(
                        smart_batch,
                        model=model,
                        batch_size=batch_size,
                        verbose=verbose
                    )
                )
            else:
                batch_enriched = embed_chunks(
                    smart_batch,
                    model=model,
                    batch_size=batch_size,  # Utiliser batch_size pour les appels API
                    verbose=verbose
                )
            enriched_chunks.extend(batch_enriched)
    else:
        # Traitement standard
        if verbose:
            num_batches = (len(all_chunks) - 1) // batch_size + 1
            logger.info(f"Taille des batchs: {batch_size}")
            logger.info(f"Vectorisation en cours ({num_batches} batch(s) à traiter)")

        # Créer les embeddings
        if use_async:
            enriched_chunks = asyncio.run(
                embed_chunks_async(
                    all_chunks,
                    model=model,
                    batch_size=batch_size,
                    verbose=verbose
                )
            )
        else:
            enriched_chunks = embed_chunks(
                all_chunks,
                model=model,
                batch_size=batch_size,
                verbose=verbose
            )

    # Réorganiser par fichier (seulement si plusieurs fichiers)
    if len(results) > 1:
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
    else:
        # Un seul fichier ou chunks combinés
        enriched_results = [{
            'file_path': results[0]['file_path'],
            'file_name': results[0]['file_name'],
            'num_chunks': len(enriched_chunks),
            'total_chars': sum(len(c.get('content', '')) for c in enriched_chunks),
            'chunks': enriched_chunks
        }]

    if verbose:
        logger.info(f"Vectorisation terminée: {len(enriched_chunks)} embeddings créés")

    return enriched_results
