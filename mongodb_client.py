"""
MongoDB client for storing document extraction results using MongoEngine and PyMongo
"""
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from mongoengine import connect, disconnect_all
import logging

from models import DocumentExtraction, ExtractionResult

logger = logging.getLogger(__name__)


class MongoDBClient:
    """Client for interacting with MongoDB using MongoEngine ODM and PyMongo"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "document_extraction",
        collection_name: str = "documents"
    ):
        """
        Initialize MongoDB client with both MongoEngine and PyMongo

        Args:
            connection_string: MongoDB connection string. If None, reads from MONGODB_URI env variable
            database_name: Name of the database to use
            collection_name: Name of the collection to use
        """
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017/"
        )
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            # Initialize MongoEngine connection
            self.mongoengine_connection = connect(
                db=self.database_name,
                host=self.connection_string,
                alias='default',
                uuidRepresentation='standard'
            )
            logger.info(f"MongoEngine connected to: {self.database_name}")

            # Initialize PyMongo client for raw operations
            self.client: MongoClient = MongoClient(self.connection_string)
            self.db: Database = self.client[self.database_name]
            self.collection: Collection = self.db[self.collection_name]

            # Test connection
            self.client.server_info()
            logger.info(f"PyMongo connected to MongoDB: {self.database_name}.{self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def save_extraction_result(
        self,
        filename: str,
        content_type: str,
        size_bytes: int,
        extraction_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save document extraction result to MongoDB using MongoEngine

        Args:
            filename: Name of the uploaded file
            content_type: MIME type of the file
            size_bytes: Size of the file in bytes
            extraction_result: Result from Gemini extraction
            metadata: Additional metadata to store

        Returns:
            Document ID as string
        """
        try:
            # Create ExtractionResult embedded document
            extraction_embedded = ExtractionResult(
                category=extraction_result.get('category'),
                main_topic=extraction_result.get('main_topic'),
                key_entities=extraction_result.get('key_entities', []),
                language=extraction_result.get('language'),
                confidence=extraction_result.get('confidence'),
                summary=extraction_result.get('summary'),
                additional_data={k: v for k, v in extraction_result.items()
                               if k not in ['category', 'main_topic', 'key_entities',
                                          'language', 'confidence', 'summary']}
            )

            # Create DocumentExtraction document
            document = DocumentExtraction(
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                extraction_result=extraction_embedded,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save using MongoEngine
            document.save()
            document_id = str(document.id)
            logger.info(f"Saved document to MongoDB using MongoEngine: {document_id}")
            return document_id

        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            raise

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID using MongoEngine

        Args:
            document_id: Document ID as string

        Returns:
            Document data or None if not found
        """
        try:
            document = DocumentExtraction.objects(id=document_id).first()
            if document:
                return document.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            return None

    def get_all_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all documents with pagination using MongoEngine

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            sort_by: Field to sort by
            ascending: Sort in ascending order if True, descending if False

        Returns:
            List of documents
        """
        try:
            # Build query with sorting
            query = DocumentExtraction.objects()

            # Apply sorting
            sort_field = sort_by if ascending else f"-{sort_by}"
            query = query.order_by(sort_field)

            # Apply pagination
            query = query.skip(skip).limit(limit)

            # Convert to list of dictionaries
            documents = [doc.to_dict() for doc in query]
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def search_documents(
        self,
        query_params: Dict[str, Any],
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search documents by query using MongoEngine

        Args:
            query_params: Query parameters (e.g., {'filename__contains': 'test'})
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of matching documents
        """
        try:
            # Build MongoEngine query
            query = DocumentExtraction.objects(**query_params)
            query = query.skip(skip).limit(limit)

            # Convert to list of dictionaries
            documents = [doc.to_dict() for doc in query]
            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def search_by_category(
        self,
        category: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search documents by category using MongoEngine

        Args:
            category: Category to search for
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of matching documents
        """
        try:
            query = DocumentExtraction.objects(
                extraction_result__category=category
            ).skip(skip).limit(limit)

            documents = [doc.to_dict() for doc in query]
            return documents

        except Exception as e:
            logger.error(f"Error searching by category: {str(e)}")
            return []

    def search_by_language(
        self,
        language: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search documents by language using MongoEngine

        Args:
            language: Language to search for
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of matching documents
        """
        try:
            query = DocumentExtraction.objects(
                extraction_result__language=language
            ).skip(skip).limit(limit)

            documents = [doc.to_dict() for doc in query]
            return documents

        except Exception as e:
            logger.error(f"Error searching by language: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collection statistics using both MongoEngine and PyMongo aggregation

        Returns:
            Dictionary containing statistics
        """
        try:
            # Total documents using MongoEngine
            total_documents = DocumentExtraction.objects.count()

            # Get content type distribution using PyMongo aggregation
            content_type_pipeline = [
                {"$group": {"_id": "$content_type", "count": {"$sum": 1}}}
            ]
            content_types = list(self.collection.aggregate(content_type_pipeline))

            # Get average file size using PyMongo aggregation
            avg_size_pipeline = [
                {"$group": {"_id": None, "avg_size": {"$avg": "$size_bytes"}}}
            ]
            avg_size_result = list(self.collection.aggregate(avg_size_pipeline))
            avg_size = avg_size_result[0]["avg_size"] if avg_size_result else 0

            # Get category distribution using PyMongo aggregation
            category_pipeline = [
                {"$group": {"_id": "$extraction_result.category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            categories = list(self.collection.aggregate(category_pipeline))

            # Get language distribution using PyMongo aggregation
            language_pipeline = [
                {"$group": {"_id": "$extraction_result.language", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            languages = list(self.collection.aggregate(language_pipeline))

            return {
                "total_documents": total_documents,
                "content_type_distribution": {ct["_id"]: ct["count"] for ct in content_types},
                "average_file_size_bytes": avg_size,
                "category_distribution": {cat["_id"]: cat["count"] for cat in categories if cat["_id"]},
                "language_distribution": {lang["_id"]: lang["count"] for lang in languages if lang["_id"]}
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID using MongoEngine

        Args:
            document_id: Document ID as string

        Returns:
            True if deleted, False otherwise
        """
        try:
            document = DocumentExtraction.objects(id=document_id).first()
            if document:
                document.delete()
                logger.info(f"Deleted document: {document_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a document using MongoEngine

        Args:
            document_id: Document ID as string
            updates: Dictionary of fields to update

        Returns:
            True if updated, False otherwise
        """
        try:
            document = DocumentExtraction.objects(id=document_id).first()
            if document:
                for key, value in updates.items():
                    if hasattr(document, key):
                        setattr(document, key, value)
                document.update_timestamp()
                logger.info(f"Updated document: {document_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def bulk_insert(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Bulk insert documents using MongoEngine

        Args:
            documents: List of document dictionaries

        Returns:
            List of inserted document IDs
        """
        try:
            doc_objects = []
            for doc_data in documents:
                extraction_data = doc_data.get('extraction_result', {})
                extraction_embedded = ExtractionResult(
                    category=extraction_data.get('category'),
                    main_topic=extraction_data.get('main_topic'),
                    key_entities=extraction_data.get('key_entities', []),
                    language=extraction_data.get('language'),
                    confidence=extraction_data.get('confidence'),
                    summary=extraction_data.get('summary')
                )

                doc = DocumentExtraction(
                    filename=doc_data['filename'],
                    content_type=doc_data['content_type'],
                    size_bytes=doc_data['size_bytes'],
                    extraction_result=extraction_embedded,
                    metadata=doc_data.get('metadata', {})
                )
                doc_objects.append(doc)

            # Bulk insert
            inserted_docs = DocumentExtraction.objects.insert(doc_objects)
            document_ids = [str(doc.id) for doc in inserted_docs]
            logger.info(f"Bulk inserted {len(document_ids)} documents")
            return document_ids

        except Exception as e:
            logger.error(f"Error in bulk insert: {str(e)}")
            return []

    def close(self):
        """Close MongoDB connections"""
        try:
            # Close PyMongo client
            self.client.close()
            # Disconnect MongoEngine
            disconnect_all()
            logger.info("MongoDB connections closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
