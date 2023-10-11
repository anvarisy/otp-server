# from sqlalchemy import Column, Integer, String, ForeignKey, Sequence, TIMESTAMP, Boolean, Float, Time, ForeignKeyConstraint, BigInteger, UUID
# from sqlalchemy.ext.declarative import declarative_base

# Base = declarative_base()

# class Merchant(Base):
#     __tablename__ = 'tb_merchant'
#     id = Column(Integer, Sequence('tb_merchant_id_seq'), primary_key=True)
#     merchant_name = Column(String, nullable=False)

# class Provider(Base):
#     __tablename__ = 'tb_provider'
#     id = Column(Integer, Sequence('tb_provider_id_seq'), primary_key=True)
#     provider_name = Column(String, nullable=False)
#     price = Column(BigInteger, nullable=False)
    
# class OtpTransaction(Base):
#     __tablename__ = 'tb_otp_transaction'
    
#     otp_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
#     merchant_id = Column(Integer, ForeignKey('tb_merchant.id'))
#     provider_id = Column(Integer, ForeignKey('tb_provider.id'), nullable=False)
    
#     type = Column(String, nullable=False)
#     pic_id = Column(String, nullable=False)
#     otp = Column(String, nullable=False)
#     timestamp = Column(TIMESTAMP, nullable=False)
#     purpose = Column(String, nullable=False)
    
#     latitude = Column(Float)
#     longitude = Column(Float)
    
#     device_name = Column(String)
#     os_version = Column(String)
#     manufacturer = Column(String)
#     cpu_info = Column(String)
#     platform = Column(String)
#     ip = Column(String)
    
#     is_active = Column(Boolean)
#     expired_at = Column(TIMESTAMP)
#     updated_at = Column(Time)
#     created_at = Column(Time)
    
#     __table_args__ = (
#         ForeignKeyConstraint(['merchant_id'], ['tb_merchant.id']),
#         ForeignKeyConstraint(['provider_id'], ['tb_provider.id']),
#     )